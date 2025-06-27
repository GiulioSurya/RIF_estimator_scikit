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
Date: 26/06/2025
Version: 1.3
"""

from typing import Dict, Optional, List, Tuple
import numpy as np
from numbers import Integral, Real
from skopt import BayesSearchCV
from skopt.space import Integer
from sklearn.base import BaseEstimator, TransformerMixin, clone, _fit_context
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import Interval, StrOptions
from functools import partial
import warnings
from . import DataFrameFingerprint

# Default search space for Random Forest hyperparameter optimization
_DEFAULT_RF_SPACE: Dict[str, Integer] = {
    "n_estimators": Integer(100, 500),  # Number of trees in the forest
    "max_depth": Integer(3, 30),  # Maximum depth of trees
    "min_samples_split": Integer(2, 10),  # Minimum samples to split internal node
    "min_samples_leaf": Integer(1, 10),  # Minimum samples in leaf node
}


class ResidualGenerator(TransformerMixin, BaseEstimator):
    """
    Generate leakage-free residuals for a set of target columns using numpy arrays.

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
    ind_indices : List[int]
        List of target column indices for which residuals are required.
    ind_cols_dict : Dict[int, List[int]]
        Mapping from target column indices to their environmental column indices.
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
        Number of iterations for BayesearchCV.
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
    models_ : Dict[int, RandomForestRegressor]
        Fitted models for each target column index.
    best_params_ : Dict[int, Dict]
        Best hyperparameters found for each target column index.
    ind_cols_dict : Dict[int, List[int]]
        Mapping from target column indices to their environmental column indices.

    Examples
    --------
    >>> # Basic usage with numpy array
    >>> rg = ResidualGenerator(
    ...     ind_indices=[0, 1],  # target column indices
    ...     ind_cols_dict={0: [2, 3, 4], 1: [2, 3, 4]},  # env columns for each target
    ...     strategy='oob'
    ... )
    >>> rg.fit(X_array)
    >>> residuals = rg.transform(X_array)

    >>> # Advanced usage with different environmental columns per target
    >>> rg = ResidualGenerator(
    ...     ind_indices=[0, 1],
    ...     ind_cols_dict={0: [2, 3], 1: [3, 4, 5]},  # different env cols per target
    ...     strategy='kfold',
    ...     bayes_search=True
    ... )
    >>> rg.fit(X_array)
    >>> residuals = rg.transform(X_array)

    Notes
    -----
    This class works exclusively with numpy arrays and integer column indices.
    All column references must be integer indices into the array columns.
    """

    # Scikit-learn automatic parameter validation
    _parameter_constraints = {
        "ind_indices": [list],  # Must be a list of integers
        "ind_cols_dict": [dict],  # Must be a dictionary
        "strategy": [StrOptions({"oob", "kfold"}), None],  # oob, kfold, or None
        "kfold_splits": [Interval(Integral, 2, None, closed="left")],  # >= 2
        "bayes_search": ["boolean"],
        "bayes_iter": [Interval(Integral, 1, None, closed="left")],  # >= 1
        "bayes_cv": [Interval(Integral, 2, None, closed="left")],  # >= 2
        "search_space": [dict, None],
        "rf_params": [dict, None],
        "random_state": [Interval(Integral, 0, None, closed="left"), None],  # >= 0 or None
    }

    def __init__(
            self,
            ind_indices: List[int],
            ind_cols_dict: Dict[int, List[int]],
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

        self.ind_indices = ind_indices
        self.ind_cols_dict = ind_cols_dict
        self.strategy = strategy
        self.kfold_splits = kfold_splits
        self.bayes_search = bayes_search
        self.bayes_iter = bayes_iter
        self.bayes_cv = bayes_cv
        self.search_space = search_space
        self.rf_params = rf_params
        self.random_state = random_state

        # Custom business logic validation (sklearn can't know this)
        if set(ind_indices) != set(ind_cols_dict.keys()):
            raise ValueError("ind_indices must match the keys of ind_cols_dict")

        # Internal attributes filled during fit
        self.models_: Dict[int, RandomForestRegressor]
        self.best_params_: Dict[int, Dict]
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
            self, X_env: np.ndarray, y_ind: np.ndarray
    ) -> Dict[str, int]:
        """
        Perform Bayesian hyperparameter optimization using BayesSearchCV.

        This method uses Bayesian optimization to find optimal hyperparameters
        for the Random Forest model. It's more efficient than grid search,
        especially with large parameter spaces.

        Parameters
        ----------
        X_env : np.ndarray
            Environmental features for prediction.
        y_ind : np.ndarray
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

    def _fit_single_model(self, X_env: np.ndarray, y_ind: np.ndarray, params: dict) -> RandomForestRegressor.fit:
        """
        Fit a single RandomForest model with given parameters.

        This method is designed to be cacheable for performance optimization.
        It combines the strategy-specific configuration with the provided
        hyperparameters to create and fit a Random Forest model.

        Parameters
        ----------
        X_env : np.ndarray
            Environmental features for training.
        y_ind : np.ndarray
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
        if np.isnan(residuals).any():
            warnings.warn(
                "OOB predictions contain NaN. Residuals will still be computed, "
                "but it is recommended to use strategy='kfold' instead of 'oob' to avoid this issue.",
                UserWarning
            )

        return residuals

    def _get_prediction_kfold(
            self, model: RandomForestRegressor, X_env: np.ndarray, y_col: np.ndarray
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
        X_env : np.ndarray
            Environmental features.
        y_col : np.ndarray
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
            self, model: RandomForestRegressor, X_env: np.ndarray
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
        X_env : np.ndarray
            Environmental features.

        Returns
        -------
        np.ndarray
            Model predictions.
        """
        return model.predict(X_env)

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X: np.ndarray, y=None) -> "ResidualGenerator":
        """
        Train one Random Forest model per target column.

        This method fits a separate Random Forest model for each target column,
        optionally performing hyperparameter optimization. The fitted models
        are stored internally for later use in transformation.

        Parameters
        ----------
        X : np.ndarray
            Training data containing both target and environmental columns.
            Should already be validated and converted by RIF.
        y : ignored
            Not used, present for API compatibility.

        Returns
        -------
        ResidualGenerator
            Fitted estimator.
        """
        # sklearn automatically calls self._validate_params() via @_fit_context decorator

        # Store fingerprint of original data
        self._training_data_fingerprint_ = DataFrameFingerprint(X)

        # Initialize storage for models and parameters
        self.models_ = {}
        self.best_params_ = {}

        # Fit models for each target column
        for target_idx in self.ind_indices:
            # Get indices for this target's environmental columns
            env_indices = self.ind_cols_dict[target_idx]

            # Extract data using indices
            X_env = X[:, env_indices]
            y_ind = X[:, target_idx]

            # Handle hyperparameter optimization
            if self.bayes_search:
                params = self._bayesian_search(X_env, y_ind)
            else:
                params = self.rf_params if self.rf_params is not None else {}

            self.models_[target_idx] = self._fit_single_model(X_env, y_ind, params)
            self.best_params_[target_idx] = params

        self._residual_cache_.clear()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute residuals for the input data.

        Results are cached by data fingerprint, so subsequent calls with
        data that have identical structure and content are O(1).

        The method automatically detects whether the input data is the same
        as the training data and applies the appropriate prediction strategy
        to avoid data leakage.

        Parameters
        ----------
        X : np.ndarray
            Data for which to compute residuals.
            Should already be validated and converted by RIF.

        Returns
        -------
        np.ndarray
            Residual matrix with shape (n_samples, n_target_columns).
            Each column contains residuals for the corresponding target variable.

        Warns
        -----
        UserWarning
            If the data has the same structure as the training data but
            different content, which might indicate potential data leakage.
        """
        # Ensure the model has been fitted
        check_is_fitted(self, "models_")

        # Create fingerprint of the current data
        current_fingerprint = DataFrameFingerprint(X)
        fingerprint_hash = hash(current_fingerprint)

        # Check if residuals are already cached
        if fingerprint_hash in self._residual_cache_:
            return self._residual_cache_[fingerprint_hash]

        residuals: List[np.ndarray] = []

        # Check if this is the same data used for training
        is_training_set = current_fingerprint == self._training_data_fingerprint_

        # Compute residuals for each target column
        for target_idx in self.ind_indices:
            model = self.models_[target_idx]

            # Get indices for this target's columns
            env_indices = self.ind_cols_dict[target_idx]

            # Extract data using indices
            X_env = X[:, env_indices]
            y_col = X[:, target_idx]

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
                        "\nThe data has the same structure (shape, columns, dtypes) as the one used "
                        "during fit. If you are using the same data but modified (e.g., after transformations "
                        "or other changes), you might encounter data leakage. Avoid making structural modifications to "
                        "the data between fit and predict to ensure the validity of the residuals.",
                        UserWarning,
                        stacklevel=2
                    )

                # For new data, always use standard prediction
                preds = model.predict(X_env)

            # Calculate residuals (actual - predicted)
            residuals.append(y_col - preds)

        # Cache and return results
        result = np.column_stack(residuals).astype(float)
        self._residual_cache_[fingerprint_hash] = result
        return result