"""
Residual Isolation Forest (RIF) - Contextual Anomaly Detection Module

This module implements the Residual Isolation Forest algorithm, which combines
residual analysis with Isolation Forest for contextual anomaly detection.
The algorithm identifies anomalies by first removing environmental/contextual
effects through regression, then applying Isolation Forest on the residuals.

The main idea is that normal behavior can often be explained by environmental
factors (e.g., high CPU usage during scheduled tasks). By modeling and removing
these expected patterns, the residuals capture only the unexplained variation,
making anomalies more apparent.

Author: Giulio Surya Lo Verde
Date: 26/06/2025
Version: 1.3

References
----------
.. [1] Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008).
       "Isolation forest." In 2008 eighth ieee international conference
       on data mining (pp. 413-422). IEEE.
.. [2] Song, X., Wu, M., Jermaine, C., & Ranka, S. (2007).
       "Conditional Anomaly Detection." IEEE Transactions on Knowledge
       and Data Engineering, 19(5), 631-645.
       https://doi.org/10.1109/TKDE.2007.1009
.. [3] Angiulli, F., Fassetti, F., & Serrao, C. (2023).
       "Anomaly detection with correlation laws." Data & Knowledge
       Engineering, 145, 102181.
       https://doi.org/10.1016/j.datak.2023.102181
.. [4] Calikus, E., Nowaczyk, S., & Dikmen, O. (2025).
       "Context discovery for anomaly detection." International Journal
       of Data Science and Analytics, 19(1), 99-113.
       https://doi.org/10.1007/s41060-024-00586-x
.. [5] Shao, C., Du, X., Yu, J., & Chen, J. (2022).
       "Cluster-Based Improved Isolation Forest." Entropy, 24(5), 611.
       https://doi.org/10.3390/e24050611
.. [6] Bouman, R., Bukhsh, Z., & Heskes, T. (2023).
       "Unsupervised anomaly detection algorithms on real-world data:
       How many do we need?" arXiv preprint arXiv:2305.00735.
       https://doi.org/10.48550/ARXIV.2305.00735
.. [7] Hawkins, D. M. (1980).
       "Identification of Outliers." Springer Netherlands.
       https://doi.org/10.1007/978-94-015-3994-4
"""

from typing import Sequence, Dict, Optional, Union, List, Tuple
import numpy as np
from numbers import Integral, Real
from skopt.space import Integer
from sklearn.base import BaseEstimator, OutlierMixin, _fit_context
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval, StrOptions
from utility import ResidualGenerator, get_column_indices


class ResidualIsolationForest(OutlierMixin, BaseEstimator):
    """
    Contextual anomaly detector based on residuals and Isolation Forest.

    This estimator applies a regression model to predict individual (behavioral)
    variables (`ind_cols`) from contextual (environmental) variables (`env_cols`)
    using the `ResidualGenerator` module. The residuals from this regression are
    used as input for the Isolation Forest algorithm to detect anomalies in an
    unsupervised manner.

    The algorithm works in two stages:
    1. **Residual Generation**: Random Forest models predict target variables
       from environmental features. The prediction errors (residuals) represent
       behavior that cannot be explained by the context.
    2. **Anomaly Detection**: Isolation Forest identifies anomalies in the
       residual space, where outliers represent contextually unexpected behavior.

    Parameters
    ----------
    ind_cols : Sequence[str] or Sequence[int] or Dict[str, Sequence[str]] or Dict[int, Sequence[int]]
        Individual/behavioral columns to analyze for anomalies.
        - If Sequence of str: Names of columns representing behavioral features
          that will all use the same environmental predictors (for DataFrame input).
        - If Sequence of int: Indices of columns representing behavioral features
          that will all use the same environmental predictors (for numpy array input).
        - If Dict: Mapping from each target column to its specific
          environmental predictors, allowing different contexts per target.

    env_cols : Sequence[str] or Sequence[int], optional
        Environmental/contextual columns used as predictors.
        - Used for all ind_cols if ind_cols is a sequence.
        - Ignored if ind_cols is a dictionary (use dict values instead).

    contamination : float, default=0.10
        Expected proportion of anomalies in the dataset.
        - Should be tuned based on domain knowledge
        - Lower values make the model more selective
        - Range: (0, 0.5]

    residual_strategy : {'oob', 'kfold', None}, default='oob'
        Strategy for computing training set residuals to avoid overfitting:
        - 'oob': Out-of-bag predictions (fast, may contain NaN)
        - 'kfold': K-fold cross-validation (slower, no NaN)
        - None: Direct predictions (fastest but risk of overfitting)

    bayes_search : bool, default=False
        Whether to perform Bayesian hyperparameter optimization for Random Forest.
        - Recommended for complex non-linear relationships
        - Increases training time but may improve performance

    bayes_iter : int, default=3
        Number of iterations for Bayesian optimization.
        Higher values explore more hyperparameter combinations.

    bayes_cv : int, default=3
        Number of cross-validation folds during Bayesian optimization.

    rf_search_space : dict[str, skopt.space.Integer], optional
        Custom search space for Random Forest hyperparameters.
        Keys can include: 'n_estimators', 'max_depth', 'min_samples_split', etc.

    rf_params : dict, optional
        Fixed Random Forest parameters (used when bayes_search=False).
        Example: {'n_estimators': 100, 'max_depth': 10}

    iso_params : dict, optional
        Additional parameters for Isolation Forest.
        Example: {'max_features': 1, 'max_samples': 'auto'}

    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    generator_ : ResidualGenerator
        The fitted residual generator instance.

    if_ : IsolationForest
        The fitted Isolation Forest model.

    Examples
    --------
    >>> # Basic usage with pandas DataFrame
    >>> rif = ResidualIsolationForest(
    ...     ind_cols=['cpu_usage', 'memory_usage'],
    ...     env_cols=['time_of_day', 'day_of_week'],
    ...     contamination=0.05
    ... )
    >>> rif.fit(X_train)
    >>> anomalies = rif.predict(X_test)  # -1 for anomalies, 1 for normal

    >>> # Basic usage with numpy array
    >>> rif = ResidualIsolationForest(
    ...     ind_cols=[0, 1],  # indices of behavioral columns
    ...     env_cols=[2, 3],  # indices of environmental columns
    ...     contamination=0.05
    ... )
    >>> rif.fit(X_train)
    >>> anomalies = rif.predict(X_test)

    >>> # Advanced usage with different contexts per target
    >>> rif = ResidualIsolationForest(
    ...     ind_cols={
    ...         'cpu_usage': ['time_of_day', 'running_processes'],
    ...         'network_traffic': ['time_of_day', 'connected_users']
    ...     },
    ...     contamination=0.10,
    ...     bayes_search=True,
    ...     residual_strategy='kfold'
    ... )
    >>> rif.fit(X_train)
    >>> anomaly_scores = rif.decision_function(X_test)

    Notes
    -----
    The effectiveness of RIF depends on:
    1. **Quality of environmental variables**: They should capture factors
       that legitimately influence the target variables.
    2. **Contamination parameter**: Should reflect actual anomaly rate.
    3. **Residual strategy**: 'oob' or 'kfold' recommended for training data.

       **Important**: When applying the model to the same dataset used for
       training, maintaining data integrity is essential to preserve the
       leakage-free properties of the algorithm. The data fingerprinting
       mechanism relies on exact structural and content matching to correctly
       identify cached residuals. Any modifications to the dataset (e.g.,
       sorting, filtering, or transformations) will
       invalidate the fingerprint, causing the system to recompute residuals
       using the fitted models rather than the appropriate out-of-bag or
       cross-validated predictions. This would introduce data leakage and
       compromise the statistical validity of the results. Therefore, it is
       strongly advised to maintain the dataset in its exact original state
       between fit and predict operations when evaluating on training data.

    The algorithm assumes that:
    - Normal behavior can be partially explained by environmental factors
    - Anomalies deviate from expected patterns given the context
    - The relationship between environment and behavior can be modeled

    This class accepts both pandas DataFrames and numpy arrays as input.

    See Also
    --------
    ResidualGenerator : The module that generates contextual residuals
    sklearn.ensemble.IsolationForest : The base anomaly detection algorithm
    """

    # Scikit-learn automatic parameter validation
    _parameter_constraints = {
        "ind_cols": [dict, list, tuple],  # Can be dict or sequence
        "env_cols": [list, tuple, None],  # Sequence or None
        "contamination": [Interval(Real, 0.0, 0.5, closed="neither")],
        "residual_strategy": [StrOptions({"oob", "kfold"}), None],
        "bayes_search": ["boolean"],
        "bayes_iter": [Interval(Integral, 1, None, closed="left")],
        "bayes_cv": [Interval(Integral, 2, None, closed="left")],
        "rf_search_space": [dict, None],
        "rf_params": [dict, None],
        "iso_params": [dict, None],
        "random_state": [Interval(Integral, 0, None, closed="left"), None],  # >= 0 or None
    }

    def __init__(
            self,
            ind_cols: Union[Sequence[str], Sequence[int], Dict[str, Sequence[str]], Dict[int, Sequence[int]]],
            env_cols: Optional[Union[Sequence[str], Sequence[int]]] = None,
            *,
            contamination: float = 0.10,
            residual_strategy: str | None = "oob",
            bayes_search: bool = False,
            bayes_iter: int = 3,
            bayes_cv: int = 3,
            rf_search_space: Optional[Dict[str, Integer]] = None,
            rf_params: Optional[Dict] = None,
            iso_params: Optional[Dict] = None,
            random_state: Optional[int] = None,
    ) -> None:
        """Initialize the Residual Isolation Forest detector."""

        self.ind_cols = ind_cols
        self.env_cols = env_cols
        self.contamination = contamination
        self.residual_strategy = residual_strategy
        self.bayes_search = bayes_search
        self.bayes_iter = bayes_iter
        self.bayes_cv = bayes_cv
        self.rf_search_space = rf_search_space
        self.rf_params = rf_params
        self.iso_params = iso_params
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None):
        """
        Fit the residual generator and Isolation Forest on training data.

        This method performs the two-stage training process:
        1. Fits Random Forest models to learn environmental patterns
        2. Trains Isolation Forest on the resulting residuals

        The residuals represent the portion of behavior that cannot be
        explained by environmental factors, making anomalies more apparent.

        Parameters
        ----------
        X : array-like
            Training dataset containing both environmental and behavioral columns.
            Supports pandas DataFrames, numpy arrays, lists, tuples, and other array-like objects.
            Must include all columns specified in ind_cols and env_cols.

        y : ignored
            Not used, present for scikit-learn API compatibility.
            This is an unsupervised method.

        Returns
        -------
        self : ResidualIsolationForest
            The fitted estimator instance.

        Notes
        -----
        The quality of anomaly detection depends heavily on:
        - The predictive power of environmental variables
        - The appropriateness of the contamination parameter
        - The quality and representativeness of training data
        """
        # sklearn automatically calls self._validate_params() via @_fit_context decorator

        # FIRST: Extract column indices before any transformation
        ind_indices, ind_cols_dict = get_column_indices(
            X=X,
            ind_cols=self.ind_cols,
            env_cols=self.env_cols
        )

        # THEN: Validate input data and convert to numpy array
        X = validate_data(
            self,
            X=X,
            reset=True,
            ensure_2d=True,
            dtype=None,
            accept_sparse=False,
            estimator=self
        )

        # Initialize the residual generator with indices (not column names)
        self.generator_ = ResidualGenerator(
            ind_indices=ind_indices,
            ind_cols_dict=ind_cols_dict,
            strategy=self.residual_strategy,
            bayes_search=self.bayes_search,
            bayes_iter=self.bayes_iter,
            bayes_cv=self.bayes_cv,
            search_space=self.rf_search_space,
            rf_params=self.rf_params,
            random_state=self.random_state,
        )

        # Fit and transform using numpy array
        res_train = self.generator_.fit_transform(X)

        iso_params_to_use = self.iso_params if self.iso_params is not None else {}

        self.if_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            **iso_params_to_use,
        ).fit(res_train)

        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict whether each observation is an anomaly.

        For efficiency, if X matches the training data, cached residuals
        are automatically reused. Otherwise, residuals are computed using
        the fitted Random Forest models.

        Parameters
        ----------
        X : array-like
            Data to evaluate for anomalies. Supports pandas DataFrames, numpy arrays,
            lists, tuples, and other array-like objects.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Anomaly labels for each observation:
            - -1: Anomaly (contextually unexpected behavior)
            - +1: Normal (behavior explained by context)

        Examples
        --------
        >>> # Detect anomalies in new data
        >>> predictions = rif.predict(X_test)
        >>> anomaly_mask = predictions == -1
        >>> anomalous_samples = X_test[anomaly_mask]
        """

        X = validate_data(
            self,
            X=X,
            reset=False,  # Don't reset n_features_in_
            ensure_2d=True,
            dtype=None,
            accept_sparse=False,
            estimator=self
        )

        # Ensure the model has been fitted
        check_is_fitted(self, "if_")

        # Transform to residual space
        res = self.generator_.transform(X)

        # Predict using Isolation Forest
        return self.if_.predict(res)

    def decision_function(self, X) -> np.ndarray:
        """
        Compute anomaly scores for each observation.

        The decision function returns the anomaly score of each sample.
        The lower the score, the more anomalous the observation.
        This provides more granular information than binary predictions.

        Parameters
        ----------
        X : array-like
            Data to compute anomaly scores for. Supports pandas DataFrames, numpy arrays,
            lists, tuples, and other array-like objects.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Anomaly scores for each observation:
            - Negative scores: likely anomalies
            - Positive scores: likely normal
            - Magnitude indicates confidence

        Notes
        -----
        Scores can be used to:
        - Rank observations by anomaly likelihood
        - Apply custom thresholds different from contamination
        - Generate anomaly probability estimates

        Example
        -------
        >>> scores = rif.decision_function(X_test)
        >>> # Get top 10 most anomalous samples
        >>> top_anomalies_idx = np.argsort(scores)[:10]
        >>> top_anomalies = X_test.iloc[top_anomalies_idx]
        """
        X = validate_data(
            self,
            X=X,
            reset=False,  # Don't reset n_features_in_
            ensure_2d=True,
            dtype=None,
            accept_sparse=False,
            estimator=self
        )

        # Ensure the model has been fitted
        check_is_fitted(self, "if_")

        # Transform to residual space
        res = self.generator_.transform(X)

        # Get anomaly scores from Isolation Forest
        return self.if_.decision_function(res)

    def get_feature_mapping(self) -> Dict[int, List[int]]:
        """
        Return the mapping of target column indices to their environmental feature indices.

        This method provides transparency into which environmental variables
        are used to model each behavioral variable, useful for interpretation
        and debugging.

        Returns
        -------
        mapping : dict
            Dictionary where:
            - Keys: target column indices (behavioral variables)
            - Values: lists of environmental feature indices used as predictors

        Example
        -------
        >>> mapping = rif.get_feature_mapping()
        >>> print(mapping)
        {0: [2, 3, 4], 1: [2, 3, 4]}  # targets 0,1 use env features 2,3,4
        """
        check_is_fitted(self, "generator_")
        return self.generator_.ind_cols_dict

    def score_samples(self, X) -> np.ndarray:
        """
        Compute the anomaly score of each sample using the IsolationForest algorithm.

        This method returns the raw anomaly scores before applying the contamination
        threshold. Lower scores indicate higher anomaly likelihood.

        Parameters
        ----------
        X : array-like
            Data to compute anomaly scores for. Supports pandas DataFrames, numpy arrays,
            lists, tuples, and other array-like objects.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Raw anomaly scores for each observation.
            The lower the score, the more anomalous the observation.

        Notes
        -----
        The relationship between score_samples and decision_function is:
        decision_function(X) = score_samples(X) - offset_

        Where offset_ is the threshold used to separate inliers from outliers
        based on the contamination parameter.

        Examples
        --------
        >>> scores = rif.score_samples(X_test)
        >>> # Get the 10 most anomalous samples
        >>> most_anomalous_idx = np.argsort(scores)[:10]
        >>> most_anomalous_samples = X_test.iloc[most_anomalous_idx]

        >>> # Compare with decision_function
        >>> decision_scores = rif.decision_function(X_test)
        >>> raw_scores = rif.score_samples(X_test)
        >>> offset = rif.offset_
        >>> assert np.allclose(decision_scores, raw_scores - offset)
        """
        X = validate_data(
            self,
            X=X,
            reset=False,  # Don't reset n_features_in_
            ensure_2d=True,
            dtype=None,
            accept_sparse=False,
            estimator=self
        )

        # Ensure the model has been fitted
        check_is_fitted(self, "if_")

        # Transform to residual space (uses caching if available)
        res = self.generator_.transform(X)

        # Get raw anomaly scores from Isolation Forest
        return self.if_.score_samples(res)

    @property
    def offset_(self) -> float:
        """
        Offset used to define the decision function from raw scores.

        This property exposes the offset used by the underlying Isolation Forest
        to convert raw anomaly scores into decision function values.

        Returns
        -------
        float
            The offset value where decision_function = score_samples - offset_

        Notes
        -----
        When contamination='auto', offset = -0.5
        When contamination is specified, offset is computed to ensure the
        expected proportion of outliers have decision_function < 0
        """
        check_is_fitted(self, "if_")
        return self.if_.offset_