"""
Module for generating leakage-free residuals from environmental predictions.

This module provides the ResidualGenerator class, which fits Random Forest models
to predict target variables based on environmental/contextual features and computes
residuals. These residuals represent the variation in the target variables that
cannot be explained by the environmental context, making them useful for
anomaly detection tasks.

The module supports multiple strategies to avoid data leakage during residual
computation and includes Bayesian hyperparameter optimization capabilities.

Author: Giulio Surya Lo Verde
Date: 13/06/2025
Version: 1.0
"""

from typing import Sequence, Dict, Optional, List, Tuple, Union
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
import warnings
from ._df_fingerprint import DataFrameFingerprint

# Default search space for Random Forest hyperparameter optimization

_DEFAULT_RF_SPACE: Dict[str, Integer] = {
    "n_estimators": Integer(100, 500),  # Number of trees in the forest
    "max_depth": Integer(3, 30),  # Maximum depth of trees
    "min_samples_split": Integer(2, 10),  # Minimum samples to split internal node
    "min_samples_leaf": Integer(1, 10),  # Minimum samples in leaf node
}


class ResidualGenerator(BaseEstimator, TransformerMixin):
    """
    Generate leakage-free residuals for a set of target columns.

    This class fits Random Forest models to predict target columns based on
    environmental variables and computes residuals to remove environmental
    effects. It supports different prediction strategies to avoid data leakage
    and can perform Bayesian hyperparameter optimization.

    The core idea is to model the relationship between environmental (contextual)
    variables and individual (behavioral) variables, then use the residuals
    (prediction errors) as features for downstream tasks like anomaly detection.
    This approach helps identify anomalies that are unexpected given the context.

    Parameters
    ----------
    ind_cols : Sequence[str] or Dict[str, Sequence[str]]
        Target columns for which residuals are required.
        If Sequence: list of column names that will use the same env_cols.
        If Dict: mapping from target column to its specific environmental columns.
    env_cols : Sequence[str], optional
        Environmental (contextual) columns used as predictors for all ind_cols.
        Ignored if ind_cols is a dictionary.
    strategy : {"oob", "kfold", None}, default="oob"
        Prediction strategy to avoid data leakage:

        * ``"oob"``   – Out-of-bag predictions (fast, may contain NaN).
        * ``"kfold"`` – K-fold cross-validation predictions (slower, no NaN).
        * ``None``    – Plain model.predict() (fastest but potentially leaky).

    kfold_splits : int, default=5
        Number of folds when strategy is "kfold".
    bayes_search : bool, default=False
        Whether to perform Bayesian hyperparameter optimization.
    bayes_iter : int, default=3
        Number of iterations for BayesSearchCV.
    bayes_cv : int, default=3
        Number of CV folds for BayesSearchCV.
    search_space : dict[str, skopt.space.Dimension] or None, default=None
        Custom Random Forest hyperparameter search space.
        Falls back to _DEFAULT_RF_SPACE if None.
    rf_params : dict or None, default=None
        Additional keyword arguments for RandomForestRegressor.
    random_state : int or None, default=None
        Global random state for reproducibility.

    Attributes
    ----------
    models_ : Dict[str, RandomForestRegressor]
        Fitted models for each target column.
    best_params_ : Dict[str, Dict]
        Best hyperparameters found for each target column.
    ind_cols_dict : Dict[str, List[str]]
        Mapping from target columns to their environmental columns.

    Examples
    --------
    >>> # Basic usage with same environmental columns for all targets
    >>> rg = ResidualGenerator(
    ...     ind_cols=['target1', 'target2'],
    ...     env_cols=['env1', 'env2', 'env3'],
    ...     strategy='oob'
    ... )
    >>> rg.fit(df)
    >>> residuals = rg.transform(df)

    >>> # Advanced usage with different environmental columns per target
    >>> rg = ResidualGenerator(
    ...     ind_cols={
    ...         'target1': ['env1', 'env2'],
    ...         'target2': ['env2', 'env3', 'env4']
    ...     },
    ...     strategy='kfold',
    ...     bayes_search=True
    ... )
    >>> rg.fit(df)
    >>> residuals = rg.transform(df)
    """

    def __init__(
            self,
            ind_cols: Union[Sequence[str], Dict[str, Sequence[str]]],
            env_cols: Optional[Sequence[str]] = None,
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
        # Validate strategy parameter
        if strategy not in {"oob", "kfold", None}:
            raise ValueError("strategy must be 'oob', 'kfold' or None")


        self.ind_cols = ind_cols
        self.env_cols = env_cols
        self.strategy = strategy
        self.kfold_splits = kfold_splits
        self.bayes_search = bayes_search
        self.bayes_iter = bayes_iter
        self.bayes_cv = bayes_cv
        self.search_space = search_space
        self.rf_params = rf_params
        self.random_state = check_random_state(random_state)

        # Handle both list and dictionary cases for ind_cols
        if isinstance(ind_cols, dict):
            # Dictionary case: each target has its own environmental columns
            self.ind_cols_dict = {k: list(v) for k, v in ind_cols.items()}
            self.ind_cols_list = list(ind_cols.keys())  # Cambia nome per evitare conflitti
            if env_cols is not None:
                warnings.warn(
                    "env_cols parameter is ignored when ind_cols is a dictionary. "
                    "Use the dictionary values to specify environmental columns for each target.",
                    UserWarning
                )
        else:
            # Sequence case: all targets use the same environmental columns
            if env_cols is None:
                raise ValueError("env_cols must be provided when ind_cols is a sequence")
            self.ind_cols_list = list(ind_cols)
            self.ind_cols_dict = {col: list(env_cols) for col in self.ind_cols_list}


        # Internal attributes filled during fit
        self.models_: Dict[str, RandomForestRegressor]
        self.best_params_: Dict[str, Dict]
        self._training_data_fingerprint_: DataFrameFingerprint
        self._residual_cache_: Dict[int, np.ndarray] = {}

    @staticmethod
    def _get_rf_config(strategy: str | None) -> dict:
        """
        Return additional RandomForest configuration based on prediction strategy.

        Different strategies require different Random Forest configurations:
        - OOB strategy needs bootstrap=True to generate out-of-bag samples
        - K-fold and None strategies don't need bootstrap

        Parameters
        ----------
        strategy : str or None
            The prediction strategy ("oob", "kfold", or None).

        Returns
        -------
        dict
            Configuration parameters for RandomForestRegressor.
        """
        return {
            "oob": {"oob_score": True, "bootstrap": True},
            "kfold": {"oob_score": False, "bootstrap": False},
            None: {"oob_score": False, "bootstrap": False},
        }[strategy]

    def _bayesian_search(
            self, X_env: pd.DataFrame, y_ind: pd.Series
    ) -> Dict[str, int]:
        """
        Perform Bayesian hyperparameter optimization using BayesSearchCV.

        This method uses Bayesian optimization to find optimal hyperparameters
        for the Random Forest model. It's more efficient than grid search,
        especially with large parameter spaces.

        Parameters
        ----------
        X_env : pd.DataFrame
            Environmental features for prediction.
        y_ind : pd.Series
            Target variable to predict.

        Returns
        -------
        Dict[str, int]
            Best hyperparameters found by the optimization.
        """
        search_space_to_use = self.search_space if self.search_space is not None else _DEFAULT_RF_SPACE

        # Initialize base Random Forest model
        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)

        # Perform Bayesian optimization
        opt = BayesSearchCV(
            rf,
            search_spaces=search_space_to_use,
            n_iter=self.bayes_iter,
            cv=self.bayes_cv,
            n_jobs=-1,
            random_state=self.random_state,
            scoring="neg_mean_squared_error",  # Minimize MSE
            verbose=0,
        ).fit(X_env, y_ind)

        return opt.best_params_

    def _fit_single_model(self, X_env: pd.DataFrame, y_ind: pd.Series, params: dict) -> RandomForestRegressor.fit:
        """
        Fit a single RandomForest model with given parameters.

        This method is designed to be cacheable for performance optimization.
        It combines the strategy-specific configuration with the provided
        hyperparameters to create and fit a Random Forest model.

        Parameters
        ----------
        X_env : pd.DataFrame
            Environmental features for training.
        y_ind : pd.Series
            Target variable to predict.
        params : dict
            Hyperparameters for the RandomForestRegressor.

        Returns
        -------
        RandomForestRegressor
            Fitted Random Forest model.
        """
        # Combine strategy config, provided params, and base settings
        rf = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=-1,
            **self._get_rf_config(self.strategy),
            **params,
        )
        model = rf.fit(X_env, y_ind)
        return model

    def _get_prediction_oob(self, model: RandomForestRegressor) -> np.ndarray:
        """
        Get out-of-bag predictions from the fitted model.

        OOB predictions are made on samples that were not used in training
        individual trees, providing a form of internal cross-validation.
        This is fast but may produce NaN values for some samples.

        Parameters
        ----------
        model : RandomForestRegressor
            Fitted Random Forest model with OOB enabled.

        Returns
        -------
        np.ndarray
            Out-of-bag predictions.

        Warns
        -----
        UserWarning
            If OOB predictions contain NaN values.
        """
        residuals = model.oob_prediction_

        # Check for NaN values in OOB predictions
        try:
            assert_all_finite(residuals)
        except ValueError:
            print("Warning: OOB predictions contain NaN. Residuals will still be computed, "
                  "but it is recommended to use strategy='kfold' instead of 'oob' to avoid this issue.")

        return residuals

    def _get_prediction_kfold(
            self, model: RandomForestRegressor, X_env: pd.DataFrame, y_col: pd.Series
    ) -> np.ndarray:
        """
        Get K-fold cross-validation predictions.

        This method uses K-fold cross-validation to generate predictions
        for all samples. Each sample is predicted by a model trained on
        folds that don't contain that sample, avoiding data leakage.

        Parameters
        ----------
        model : RandomForestRegressor
            Random Forest model (will be cloned for CV).
        X_env : pd.DataFrame
            Environmental features.
        y_col : pd.Series
            Target variable.

        Returns
        -------
        np.ndarray
            Cross-validation predictions.
        """
        # Set up K-fold cross-validation
        cv = KFold(
            n_splits=self.kfold_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        # Get predictions using cross-validation
        # Each sample is predicted by models trained without it
        return cross_val_predict(clone(model), X_env, y_col, cv=cv, n_jobs=-1)

    def _get_prediction_none(
            self, model: RandomForestRegressor, X_env: pd.DataFrame
    ) -> np.ndarray:
        """
        Get standard model predictions (potentially leaky).

        This method uses the model to predict on the same data it was
        trained on, which can lead to overfitting and data leakage.
        Should only be used when leakage is not a concern.

        Parameters
        ----------
        model : RandomForestRegressor
            Fitted Random Forest model.
        X_env : pd.DataFrame
            Environmental features.

        Returns
        -------
        np.ndarray
            Model predictions.
        """
        return model.predict(X_env)

    def fit(self, X: pd.DataFrame, y=None) -> "ResidualGenerator":
        """
        Train one Random Forest model per target column.

        This method fits a separate Random Forest model for each target column,
        optionally performing hyperparameter optimization. The fitted models
        are stored internally for later use in transformation.

        Parameters
        ----------
        X : pd.DataFrame
            Training data containing both target and environmental columns.
        y : ignored
            Not used, present for API compatibility.

        Returns
        -------
        ResidualGenerator
            Fitted estimator.
        """
        # Initialize storage for models and parameters
        self.models_ = {}
        self.best_params_ = {}
        self._training_data_fingerprint_ = DataFrameFingerprint(X)

        for col in self.ind_cols_list:  # ← Usa ind_cols_list
            env_cols_for_this_target = self.ind_cols_dict[col]
            X_env = X[env_cols_for_this_target]
            y_ind = X[col]

            # Gestisci i valori None QUI
            if self.bayes_search:
                params = self._bayesian_search(X_env, y_ind)
            else:
                params = self.rf_params if self.rf_params is not None else {}

            self.models_[col] = self._fit_single_model(X_env, y_ind, params)
            self.best_params_[col] = params

        self._residual_cache_.clear()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute residuals for the input DataFrame.

        Results are cached by DataFrame fingerprint, so subsequent calls with
        DataFrames that have identical structure and content are O(1).

        The method automatically detects whether the input data is the same
        as the training data and applies the appropriate prediction strategy
        to avoid data leakage.

        Parameters
        ----------
        X : pd.DataFrame
            Data for which to compute residuals.

        Returns
        -------
        np.ndarray
            Residual matrix with shape (n_samples, n_target_columns).
            Each column contains residuals for the corresponding target variable.

        Warns
        -----
        UserWarning
            If the DataFrame has the same structure as the training data but
            different content, which might indicate potential data leakage.
        """
        # Ensure the model has been fitted
        check_is_fitted(self, "models_")

        # Create fingerprint of the current DataFrame
        current_fingerprint = DataFrameFingerprint(X)
        fingerprint_hash = hash(current_fingerprint)

        # Check if residuals are already cached
        if fingerprint_hash in self._residual_cache_:
            return self._residual_cache_[fingerprint_hash]

        residuals: List[np.ndarray] = []

        # Check if this is the same DataFrame used for training
        is_training_set = current_fingerprint == self._training_data_fingerprint_

        # Compute residuals for each target column
        for col in self.ind_cols:
            model = self.models_[col]
            y_col = X[col]

            # Use specific environmental columns for this target
            env_cols_for_this_target = self.ind_cols_dict[col]
            X_env = X[env_cols_for_this_target]

            if is_training_set:
                # Use the specified strategy for training data to avoid leakage
                # Each strategy has its own prediction method
                preds = {
                    "oob": partial(self._get_prediction_oob, model),
                    "kfold": partial(self._get_prediction_kfold, model, X_env, y_col),
                    None: partial(self._get_prediction_none, model, X_env)
                }[self.strategy]()
            else:
                # For new data, warn if structure matches but content differs
                # This could indicate unintended data leakage scenarios
                if current_fingerprint.matches_structure_only(self._training_data_fingerprint_):
                    warnings.warn(
                        "\nThe DataFrame has the same structure (shape, columns, dtypes) as the one used "
                        "during fit. If you are using the same DataFrame but modified (e.g., after reset_index() "
                        "or other changes), you might encounter data leakage. Avoid making structural modifications to "
                        "the DataFrame between fit and predict to ensure the validity of the residuals.",
                        UserWarning,
                        stacklevel=2
                    )

                # For new data, always use standard prediction
                preds = model.predict(X_env)

            # Calculate residuals (actual - predicted)
            residuals.append(y_col.to_numpy() - preds)

        # Cache and return results
        result = np.column_stack(residuals).astype(float)
        self._residual_cache_[fingerprint_hash] = result
        return result