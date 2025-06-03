from typing import Sequence, Dict, Optional, List
import numpy as np
import pandas as pd
import hashlib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state
from skopt import BayesSearchCV
from skopt.space import Integer

_DEFAULT_RF_SPACE: Dict[str, Integer] = {
    "n_estimators": Integer(100, 500),
    "max_depth": Integer(3, 30),
    "min_samples_split": Integer(2, 10),
    "min_samples_leaf": Integer(1, 10),
}


class OOBStrategy:
    """Strategia Out-Of-Bag."""

    def get_predictions(self, rf: RandomForestRegressor, X_env: pd.DataFrame,
                        y_ind: pd.Series, cv_splitter=None) -> tuple[RandomForestRegressor, np.ndarray]:
        rf_fit = rf.fit(X_env, y_ind)
        return rf_fit, rf_fit.oob_prediction_


class KFoldStrategy:
    """Strategia K-Fold Cross Validation."""

    def get_predictions(self, rf: RandomForestRegressor, X_env: pd.DataFrame,
                        y_ind: pd.Series, cv_splitter) -> tuple[RandomForestRegressor, np.ndarray]:
        preds = cross_val_predict(rf, X_env, y_ind, cv=cv_splitter, n_jobs=-1)
        rf_fit = rf.fit(X_env, y_ind)
        return rf_fit, preds


class NoneStrategy:
    """Strategia semplice: solo fit, nessuna predizione leakage-free."""

    def get_predictions(self, rf: RandomForestRegressor, X_env: pd.DataFrame,
                        y_ind: pd.Series, cv_splitter=None) -> tuple[RandomForestRegressor, np.ndarray]:
        rf_fit = rf.fit(X_env, y_ind)
        preds = rf_fit.predict(X_env)
        return rf_fit, preds


class ResidualGenerator(BaseEstimator, TransformerMixin):
    """
    ResidualGenerator con Strategy Pattern
    -------------------------------------
    Supporta tre strategie:
    - "oob": Out-Of-Bag predictions (leakage-free ma può avere NaN)
    - "kfold": K-fold cross-validation (leakage-free, più lento)
    - "none": Predizioni standard (veloce ma con potenziale data leakage)
    """

    def __init__(
            self,
            ind_cols: Sequence[str],
            env_cols: Sequence[str],
            *,
            strategy: str = None,
            kfold_splits: int = 5,
            bayes_search: bool = False,
            bayes_iter: int = 3,
            bayes_cv: int = 3,
            search_space: Optional[Dict[str, Integer]] = None,
            rf_params: Optional[Dict] = None,
            random_state: Optional[int] = None,
    ):
        if strategy not in {"oob", "kfold", None}:
            raise ValueError("strategy must be 'oob', 'kfold', or None")

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

        # Strategy pattern
        self._strategies = {
            "oob": OOBStrategy(),
            "kfold": KFoldStrategy(),
            None: NoneStrategy()
        }

    def _get_rf_config(self, strategy: str) -> dict:
        """Restituisce la configurazione RF per la strategia."""
        configs = {
            "oob": {"oob_score": True, "bootstrap": True},
            "kfold": {"oob_score": False, "bootstrap": False},
            None: {"oob_score": False, "bootstrap": False}
        }
        return configs[strategy]

    def _compute_dataframe_hash(self, X: pd.DataFrame) -> str:
        """Calcola un hash MD5 del DataFrame per identificarlo univocamente."""
        relevant_cols = self.ind_cols + self.env_cols
        X_relevant = X[relevant_cols]
        data_bytes = pd.util.hash_pandas_object(X_relevant, index=True).values.tobytes()
        return hashlib.md5(data_bytes).hexdigest()

    def _bayesian_search(self, X_env: pd.DataFrame, y_ind: pd.Series) -> Dict[str, int]:
        """Restituisce i best_params trovati con BayesSearchCV."""
        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        opt = BayesSearchCV(
            rf,
            search_spaces=self.search_space,
            n_iter=self.bayes_iter,
            cv=self.bayes_cv,
            n_jobs=-1,
            random_state=self.random_state,
            scoring="neg_mean_squared_error",
            verbose=1,
        ).fit(X_env, y_ind)
        return opt.best_params_

    def fit(self, X: pd.DataFrame, y=None) -> "ResidualGenerator":
        """Addestra le RF usando la strategia selezionata."""

        self.models_: Dict[str, object] = {}
        self.best_params_: Dict[str, Dict] = {}
        train_residuals: List[np.ndarray] = []

        self.training_data_hash_ = self._compute_dataframe_hash(X)
        X_env = X[self.env_cols]

        # Prepara CV splitter se necessario
        cv_splitter = None
        if self.strategy == "kfold":
            cv_splitter = KFold(
                n_splits=self.kfold_splits,
                shuffle=True,
                random_state=self.random_state
            )

        # Seleziona la strategia
        strategy_impl = self._strategies[self.strategy]
        rf_config = self._get_rf_config(self.strategy)

        for col in self.ind_cols:
            y_ind = X[col]

            # Ottieni parametri RF
            params = (
                self._bayesian_search(X_env, y_ind)
                if self.bayes_search
                else self.rf_params
            )

            # Crea RF con configurazione appropriata
            rf = RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                **rf_config,
                **params,
            )

            # Usa la strategia per ottenere predizioni
            rf_fit, preds = strategy_impl.get_predictions(rf, X_env, y_ind, cv_splitter)

            # Salva risultati
            self.models_[col] = rf_fit
            self.best_params_[col] = params
            train_residuals.append((y_ind - preds).to_numpy())

        self.train_residuals_ = np.column_stack(train_residuals).astype(float)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Calcola la matrice residui per il DataFrame X."""
        check_is_fitted(self, "models_")

        current_hash = self._compute_dataframe_hash(X)
        if current_hash == self.training_data_hash_:
            return self.train_residuals_

        X_env = X[self.env_cols]
        res = np.column_stack(
            [
                X[col].to_numpy() - self.models_[col].predict(X_env)
                for col in self.ind_cols
            ]
        )
        return res.astype(float)

    def fit_transform(self, X: pd.DataFrame, y=None):
        """
        Fit + transform, garantendo che i residui restituiti
        siano ESATTAMENTE quelli che verranno riconsegnati da
        transform(X) in chiamate successive.

        In pratica:
        1. fit() salva train_residuals_ e training_data_hash_
        2. transform(X) rileva che l'hash coincide
           e restituisce gli stessi residui (senza ricalcolarli)
        """
        self.fit(X, y)
        return self.transform(X)
