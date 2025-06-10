from typing import Sequence, Dict, Optional, List
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Integer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.utils.validation import check_is_fitted, assert_all_finite
from sklearn.utils import check_random_state
from functools import partial


_DEFAULT_RF_SPACE: Dict[str, Integer] = {
    "n_estimators": Integer(100, 500),
    "max_depth": Integer(3, 30),
    "min_samples_split": Integer(2, 10),
    "min_samples_leaf": Integer(1, 10),
}

# def hash_df(df: pd.DataFrame) -> str:
#     """Create a unique hash for a DataFrame's content."""
#     return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()




class ResidualGenerator(BaseEstimator, TransformerMixin):
    """Generate leakage‑free residuals for a set of target columns.

    Parameters
    ----------
    ind_cols : Sequence[str]
        Columns whose residuals are required.
    env_cols : Sequence[str]
        Contextual (environment) columns used as regressors.
    strategy : {"oob", "kfold", None}, default="oob"
        * ``"oob"``   – out‑of‑bag predictions (fast, may contain NaN).
        * ``"kfold"`` – K‑fold CV predictions (slower, no NaN).
        * ``None``    – plain ``model.predict`` (cheap but leaky).
    kfold_splits : int, default=5
        Number of folds when *strategy* is "kfold".
    bayes_search : bool, default=False
        Whether to perform a Bayesian hyper‑parameter search.
    bayes_iter, bayes_cv : int, default=3
        Iterations / CV folds for :class:`BayesSearchCV`.
    search_space : dict[str, skopt.space.Dimension] | None, default=None
        Custom RF search space.  Falls back to ``_DEFAULT_RF_SPACE``.
    rf_params : dict | None, default=None
        Extra keyword args for :class:`RandomForestRegressor`.
    random_state : int | None, default=None
        Global random state.
    """

    def __init__(
        self,
        ind_cols: Sequence[str],
        env_cols: Sequence[str],
        *,
        strategy: str | None = "oob",
        kfold_splits: int = 5,
        bayes_search: bool = False,
        bayes_iter: int = 3,
        bayes_cv: int = 3,
        search_space: Optional[Dict[str, Integer]] = None,
        rf_params: Optional[Dict] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if strategy not in {"oob", "kfold", None}:
            raise ValueError("strategy must be 'oob', 'kfold' or None")

        self.ind_cols = list(ind_cols)
        self.env_cols = list(env_cols)
        self.strategy = strategy
        self.kfold_splits = kfold_splits
        self.bayes_search = bayes_search
        self.bayes_iter = bayes_iter
        self.bayes_cv = bayes_cv
        self.search_space = search_space or _DEFAULT_RF_SPACE
        self.rf_params = rf_params or {}
        self.random_state = check_random_state(random_state)

        # Internal attributes filled during fit ----------------------------
        self.models_: Dict[str, RandomForestRegressor]
        self.best_params_: Dict[str, Dict]
        self._training_data_id_: int
        self._residual_cache_: Dict[int, np.ndarray] = {}


    @staticmethod
    def _get_rf_config(strategy: str | None) -> dict:
        """Return additional RF keyword args depending on *strategy*."""
        return {
            "oob": {"oob_score": True, "bootstrap": True},
            "kfold": {"oob_score": False, "bootstrap": False},
            None: {"oob_score": False, "bootstrap": False},
        }[strategy]


    def _bayesian_search(
        self, X_env: pd.DataFrame, y_ind: pd.Series
    ) -> Dict[str, int]:
        """Run BayesSearchCV and return the best parameters found."""
        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        opt = BayesSearchCV(
            rf,
            search_spaces=self.search_space,
            n_iter=self.bayes_iter,
            cv=self.bayes_cv,
            n_jobs=-1,
            random_state=self.random_state,
            scoring="neg_mean_squared_error",
            verbose=0,
        ).fit(X_env, y_ind)
        return opt.best_params_

    def _fit_single_model(self, X_env: pd.DataFrame, y_ind: pd.Series, params: dict) -> RandomForestRegressor.fit:
        """Fit a single RandomForest model - this method will be cached."""
        rf = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=-1,
            **self._get_rf_config(self.strategy),
            **params,
        )
        return rf.fit(X_env, y_ind)

    def _get_prediction_oob(self, model):
        residuals = model.oob_prediction_
        try:
            assert_all_finite(residuals)
        except ValueError:
            print("Warning: OOB predictions contain NaN. Residuals will still be computed, "
                  "but it is recommended to use strategy='kfold' instead of 'oob' to avoid this issue.")
        return residuals

    def _get_prediction_kfold(self, model, X_env, y_col):
        cv = KFold(
            n_splits=self.kfold_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        return cross_val_predict(clone(model), X_env, y_col, cv=cv, n_jobs=-1)

    def _get_prediction_none(self, model, X_env):
        return model.predict(X_env)




    def fit(self, X: pd.DataFrame, y=None) -> "ResidualGenerator":
        """Train one Random‑Forest per *independent* column."""
        self.models_ = {}
        self.best_params_ = {}

        self._training_data_id_ = id(X)

        X_env = X[self.env_cols]

        for col in self.ind_cols:
            y_ind = X[col]

            # Optional hyper‑parameter search
            params = (
                self._bayesian_search(X_env, y_ind)
                if self.bayes_search
                else self.rf_params
            )

            self.models_[col] = self._fit_single_model(X_env, y_ind, params)
            self.best_params_[col] = params

        # Reset cache (useful when refitting inside a pipeline)
        self._residual_cache_.clear()

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Return the residual matrix for X .

        Results are cached by DataFrame *identity* so subsequent calls with
        the **same object** are O(1).
        """
        check_is_fitted(self, "models_")

        key = id(X)

        #a.equals(b)

        if key in self._residual_cache_:
            return self._residual_cache_[key]

        X_env = X[self.env_cols]
        residuals: List[np.ndarray] = []
        is_training_set = key == getattr(self, "_training_data_id_", -1)

        for col in self.ind_cols:
            model = self.models_[col]
            y_col = X[col]

            # per domani, questo proviamo ad eliminarlo e faro con dizionario + metodi per il predict
            if is_training_set:
                preds = {
                    "oob": partial(self._get_prediction_oob,model),
                    "kfold": partial(self._get_prediction_kfold, model, X_env, y_col),
                    None: partial(self._get_prediction_none,model, X_env)
                }[self.strategy]()

            else:
                # Standard prediction for *new* data -------------------
                preds = model.predict(X_env)

            residuals.append(y_col.to_numpy() - preds)

        self._residual_cache_[key] = np.column_stack(residuals).astype(float)
        return self._residual_cache_[key]

git