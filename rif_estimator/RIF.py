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
Date: 13/06/2025
Version: 1.0

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

from typing import Sequence, Dict, Optional, Union
import numpy as np
import pandas as pd
from skopt.space import Integer
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted
from utility import ResidualGenerator

class ResidualIsolationForest(BaseEstimator, OutlierMixin):
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
    ind_cols : Sequence[str] or Dict[str, Sequence[str]]
        Individual/behavioral columns to analyze for anomalies.
        - If Sequence: Names of columns representing behavioral features
          that will all use the same environmental predictors.
        - If Dict: Mapping from each target column to its specific
          environmental predictors, allowing different contexts per target.

    env_cols : Sequence[str], optional
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
    generator : ResidualGenerator
        The fitted residual generator instance.

    if_ : IsolationForest
        The fitted Isolation Forest model.

    Examples
    --------
    >>> # Basic usage with uniform environmental context
    >>> rif = ResidualIsolationForest(
    ...     ind_cols=['cpu_usage', 'memory_usage'],
    ...     env_cols=['time_of_day', 'day_of_week'],
    ...     contamination=0.05
    ... )
    >>> rif.fit(X_train)
    >>> anomalies = rif.predict(X_test)  # -1 for anomalies, 1 for normal

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
       leakage-free properties of the algorithm. The DataFrame fingerprinting
       mechanism relies on exact structural and content matching to correctly
       identify cached residuals. Any modifications to the dataset (e.g.,
       index operations, sorting, filtering, or column transformations) will
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

    See Also
    --------
    ResidualGenerator : The module that generates contextual residuals
    sklearn.ensemble.IsolationForest : The base anomaly detection algorithm
    """

    def __init__(
            self,
            ind_cols: Union[Sequence[str], Dict[str, Sequence[str]]],
            env_cols: Optional[Sequence[str]] = None,
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
        self.iso_params = iso_params or {}
        self.random_state = random_state

        # Initialize the residual generator with all relevant parameters
        self.generator = ResidualGenerator(
            ind_cols=ind_cols,
            env_cols=env_cols,
            strategy=residual_strategy,
            bayes_search=bayes_search,
            bayes_iter=bayes_iter,
            bayes_cv=bayes_cv,
            search_space=rf_search_space,
            rf_params=rf_params,
            random_state=random_state,
        )


    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the residual generator and Isolation Forest on training data.

        This method performs the two-stage training process:
        1. Fits Random Forest models to learn environmental patterns
        2. Trains Isolation Forest on the resulting residuals

        The residuals represent the portion of behavior that cannot be
        explained by environmental factors, making anomalies more apparent.

        Parameters
        ----------
        X : pd.DataFrame
            Training dataset containing both environmental and behavioral columns.
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
        # Step 1: Generate residuals using the configured strategy
        # This removes the predictable (environmental) component
        res_train = self.generator.fit_transform(X)

        # Step 2: Train Isolation Forest on residuals
        # Anomalies in residual space = contextually unexpected behavior
        self.if_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            **self.iso_params,
        ).fit(res_train)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict whether each observation is an anomaly.

        For efficiency, if X matches the training data, cached residuals
        are automatically reused. Otherwise, residuals are computed using
        the fitted Random Forest models.

        Parameters
        ----------
        X : pd.DataFrame
            Data to evaluate for anomalies.
            Must contain the same columns as the training data.

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
        # Ensure the model has been fitted
        check_is_fitted(self, "if_")

        # Transform to residual space (uses caching if available)
        res = self.generator.transform(X)

        # Predict using Isolation Forest
        return self.if_.predict(res)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores for each observation.

        The decision function returns the anomaly score of each sample.
        The lower the score, the more anomalous the observation.
        This provides more granular information than binary predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Data to compute anomaly scores for.

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
        # Ensure the model has been fitted
        check_is_fitted(self, "if_")

        # Transform to residual space
        res = self.generator.transform(X)

        # Get anomaly scores from Isolation Forest
        return self.if_.decision_function(res)

    def get_feature_mapping(self) -> Dict[str, list]:
        """
        Return the mapping of target columns to their environmental features.

        This method provides transparency into which environmental variables
        are used to model each behavioral variable, useful for interpretation
        and debugging.

        Returns
        -------
        mapping : dict
            Dictionary where:
            - Keys: target column names (behavioral variables)
            - Values: lists of environmental feature names used as predictors

        Example
        -------
        >>> mapping = rif.get_feature_mapping()
        >>> print(mapping)
        {'cpu_usage': ['time_of_day', 'running_processes'],
         'memory_usage': ['time_of_day', 'running_processes']}
        """
        return self.generator.ind_cols_dict




# if __name__ == "__main__":
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
#
#     df = pd.read_csv(r"C:\Users\loverdegiulio\PycharmProjects\tesi\datas\synthetic_data_testX.csv")
#     #df = pd.read_csv(r"C:\Users\loverdegiulio\PycharmProjects\tesi\datas\synthetic_data_moderate.csv")
#
#     ENV_COLS = ["env_X0", "env_X1", "env_X2", "env_X3", "env_X4", "env_X5"]
#     IND_COLS = ["ind_Y0", "ind_Y1", "ind_Y2"]
#
#     y = df["is_anomaly"].to_numpy()
#     X = df.drop(columns=["is_anomaly", "is_outlier"])
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y,
#         test_size=0.30,
#         random_state=42,
#     )
#
#
#
#     rif = ResidualIsolationForest(
#         ind_cols=IND_COLS,
#         env_cols=ENV_COLS,
#         contamination=0.20,
#         random_state=42,
#         residual_strategy="oob",
#         bayes_search=False,
#         iso_params={"max_features": 1}
#     )
#     rif.fit(X_train)
#
#
#     splits = zip([X_train, X_test], [y_train, y_test])
#     from sklearn.metrics import precision_recall_curve, precision_score
#
#     import matplotlib
#
#     matplotlib.use("TkAgg")
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     plt.close("all")
#
#     plt.ion()
#
#     for x, y in splits:
#         # Residual Isolation Forest
#         y_pred_rif = np.where(rif.predict(x) == -1, 1, 0)
#         acc_rif = accuracy_score(y, y_pred_rif)
#         rec_rif = recall_score(y, y_pred_rif)
#         prec_rif = precision_score(y, y_pred_rif)
#         cm_rif = confusion_matrix(y, y_pred_rif)
#
#         # Isolation Forest vanilla
#         iso = IsolationForest(
#             contamination=0.10,
#             random_state=42,
#             max_features=1
#         ).fit(X_train)
#
#         y_pred_iso = np.where(iso.predict(x) == -1, 1, 0)
#         acc_iso = accuracy_score(y, y_pred_iso)
#         rec_iso = recall_score(y, y_pred_iso)
#         prec_iso = precision_score(y, y_pred_iso)
#         cm_iso = confusion_matrix(y, y_pred_iso)
#
#         # Curva Precision-Recall per RIF con thresholds
#         scores_rif = rif.decision_function(x)
#         precision_rif, recall_rif, thresholds_rif = precision_recall_curve(y, -scores_rif)
#
#         # Curva Precision-Recall per ISO con thresholds
#         scores_iso = iso.decision_function(x)
#         precision_iso, recall_iso, thresholds_iso = precision_recall_curve(y, -scores_iso)
#
#         # Calcolo F1 score per trovare soglia ottimale
#         f1_scores_rif = 2 * (precision_rif[:-1] * recall_rif[:-1]) / (precision_rif[:-1] + recall_rif[:-1] + 1e-10)
#         best_idx_rif = np.argmax(f1_scores_rif)
#         best_threshold_rif = thresholds_rif[best_idx_rif]
#
#         f1_scores_iso = 2 * (precision_iso[:-1] * recall_iso[:-1]) / (precision_iso[:-1] + recall_iso[:-1] + 1e-10)
#         best_idx_iso = np.argmax(f1_scores_iso)
#         best_threshold_iso = thresholds_iso[best_idx_iso]
#
#         # Visualizzazione
#         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
#
#         # Grafico Precision-Recall
#         ax1.plot(recall_rif, precision_rif, marker='.', linestyle='-', color='blue', label='Residual Isolation Forest')
#         ax1.plot(recall_iso, precision_iso, marker='.', linestyle='-', color='red', label='Isolation Forest')
#         # Aggiungi punto ottimale sulla curva
#         ax1.scatter(recall_rif[best_idx_rif], precision_rif[best_idx_rif], color='blue', s=100, marker='o',
#                     label=f'RIF Optimal (t={best_threshold_rif:.2f})')
#         ax1.scatter(recall_iso[best_idx_iso], precision_iso[best_idx_iso], color='red', s=100, marker='o',
#                     label=f'ISO Optimal (t={best_threshold_iso:.2f})')
#
#         ax1.set_xlabel('Recall')
#         ax1.set_ylabel('Precision')
#         ax1.set_title('Precision-Recall Curve with Optimal Thresholds')
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
#
#         # Grafico Thresholds vs F1 score
#         threshold_indices_rif = np.arange(len(thresholds_rif))
#         threshold_indices_iso = np.arange(len(thresholds_iso))
#
#         ax2.plot(thresholds_rif, f1_scores_rif, marker='.', linestyle='-', color='blue', label='RIF F1 Score')
#         ax2.plot(thresholds_iso, f1_scores_iso, marker='.', linestyle='-', color='red', label='ISO F1 Score')
#         ax2.axvline(x=best_threshold_rif, color='blue', linestyle='--',
#                     label=f'RIF Best Threshold: {best_threshold_rif:.2f}')
#         ax2.axvline(x=best_threshold_iso, color='red', linestyle='--',
#                     label=f'ISO Best Threshold: {best_threshold_iso:.2f}')
#
#         ax2.set_xlabel('Threshold')
#         ax2.set_ylabel('F1 Score')
#         ax2.set_title('F1 Score vs Threshold')
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
#
#         # Confronto metriche
#         ax3.bar(['Accuracy', 'Recall', 'Precision'],
#                 [acc_rif, rec_rif, prec_rif],
#                 width=0.4,
#                 label='RIF',
#                 color='blue',
#                 alpha=0.7)
#         ax3.bar(['Accuracy', 'Recall', 'Precision'],
#                 [acc_iso, rec_iso, prec_iso],
#                 width=0.4,
#                 label='ISO',
#                 color='red',
#                 alpha=0.7,
#                 align='edge')
#         ax3.set_ylim(0, 1)
#         ax3.set_title('Performance Metrics')
#         ax3.legend()
#         ax3.grid(True, alpha=0.3)
#
#         plt.tight_layout()
#         plt.show()
#
#         # Applicazione della soglia ottimale
#         y_pred_rif_optimal = np.where(-scores_rif >= best_threshold_rif, 1, 0)
#         y_pred_iso_optimal = np.where(-scores_iso >= best_threshold_iso, 1, 0)
#
#         # Metriche con soglia ottimale
#         acc_rif_opt = accuracy_score(y, y_pred_rif_optimal)
#         rec_rif_opt = recall_score(y, y_pred_rif_optimal)
#         prec_rif_opt = precision_score(y, y_pred_rif_optimal)
#
#         acc_iso_opt = accuracy_score(y, y_pred_iso_optimal)
#         rec_iso_opt = recall_score(y, y_pred_iso_optimal)
#         prec_iso_opt = precision_score(y, y_pred_iso_optimal)
#
#         # Stampa risultati
#         print("\n=== Residual Isolation Forest (Default Threshold) ===")
#         print(f"Accuracy : {acc_rif:.3f}")
#         print(f"Recall   : {rec_rif:.3f}")
#         print(f"Precision: {prec_rif:.3f}")
#         print("Confusion matrix:")
#         print(cm_rif)
#
#         print("\n=== Residual Isolation Forest (Optimal Threshold: {:.3f}) ===".format(best_threshold_rif))
#         print(f"Accuracy : {acc_rif_opt:.3f}")
#         print(f"Recall   : {rec_rif_opt:.3f}")
#         print(f"Precision: {prec_rif_opt:.3f}")
#         print("Confusion matrix:")
#         print(confusion_matrix(y, y_pred_rif_optimal))
#
#         print("\n=== Isolation Forest (Default Threshold) ===")
#         print(f"Accuracy : {acc_iso:.3f}")
#         print(f"Recall   : {rec_iso:.3f}")
#         print(f"Precision: {prec_iso:.3f}")
#         print("Confusion matrix:")
#         print(cm_iso)
#
#         print("\n=== Isolation Forest (Optimal Threshold: {:.3f}) ===".format(best_threshold_iso))
#         print(f"Accuracy : {acc_iso_opt:.3f}")
#         print(f"Recall   : {rec_iso_opt:.3f}")
#         print(f"Precision: {prec_iso_opt:.3f}")
#         print("Confusion matrix:")
#         print(confusion_matrix(y, y_pred_iso_optimal))
# #######################

# if __name__ == "__main__":
#
#     import numpy as np
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
#
#
#
#     env_cols = ["year", "month", "day", "latitude", "longitude"]
#     ind_cols = ["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"]
#
#
#     prepared_df = pd.read_csv(r"C:\Users\loverdegiulio\PycharmProjects\tesi\datas\elnino_prepared.csv")
#
#     # Target e feature set
#     y = prepared_df["is_anomaly"].to_numpy()
#     X = prepared_df.drop(columns=["is_anomaly", "is_outlier"])
#
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y,
#         test_size=0.30,
#         random_state=42,
#     )
#
#     from _residual_gen import global_r2
#
#     # Residual Isolation Forest
#     rif = ResidualIsolationForest(
#         ind_cols=ind_cols,
#         env_cols=env_cols,
#         contamination=0.10,
#         random_state=42,
#         residual_strategy="kfold",
#         bayes_search=False,
#         iso_params={"max_features": 1}
#     )
#     rif.fit(X_train)
#
#     splits = zip([X_train, X_test], [y_train, y_test])
#
#     for x, y in splits:
#
#         y_pred_rif = np.where(rif.predict(x) == -1, 1, 0)
#         acc_rif = accuracy_score(y, y_pred_rif)
#         rec_rif = recall_score(y, y_pred_rif)
#         cm_rif = confusion_matrix(y, y_pred_rif)
#
#         print("=== Residual Isolation Forest ===")
#         print(f"Accuracy : {acc_rif:.3f}")
#         print(f"Recall   : {rec_rif:.3f}")
#         print("Confusion matrix:\n", cm_rif)
#
#         # Isolation Forest vanilla
#         iso = IsolationForest(
#             contamination=0.10,
#             random_state=42,
#             max_features=1
#         ).fit(X_train)
#
#         iso_pred = np.where(iso.predict(x) == -1, 1, 0)
#         acc_iso = accuracy_score(y, iso_pred)
#         rec_iso = recall_score(y, iso_pred)
#         cm_iso = confusion_matrix(y, iso_pred)
#
#         print("\n=== Isolation Forest (vanilla) ===")
#         print(f"Accuracy : {acc_iso:.3f}")
#         print(f"Recall   : {rec_iso:.3f}")
#         print("Confusion matrix:\n", cm_iso)
#
#

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from rif_estimator import ResidualIsolationForest

    df = pd.read_csv(r"C:\Users\loverdegiulio\PycharmProjects\RIF_estimator_scikit\test_eif\datas\dati.csv")

    ENV_COLS = ["rate_receive_lo", "rate_transmit_lo", "rate_receive_ens33", "rate_transmit_ens33", "hour"]
    IND_COLS = ["cpu_busy", "ram_busy", "swap_busy", "disk_busy"]

    y = df["anomaly"].to_numpy()

    X = df.drop(columns=["anomaly", "time", "date"])

    # Feature da standardizzare
    feature_cols = ENV_COLS + IND_COLS
    X_raw = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    #X_scaled_df["date"] = df["date"]
    X_scaled_df["hour"] = df["hour"]

    feature_mapping = {
        "cpu_busy": ["rate_receive_lo", "rate_transmit_lo", "rate_receive_ens33", "rate_transmit_ens33", "ram_busy", "swap_busy"],
        "ram_busy": ["rate_receive_lo", "rate_transmit_lo", "rate_receive_ens33", "rate_transmit_ens33"],
        "swap_busy":["rate_receive_lo", "rate_transmit_lo", "rate_receive_ens33", "rate_transmit_ens33", "ram_busy"],
        "disk_busy": ["rate_receive_lo", "rate_transmit_lo", "rate_receive_ens33", "rate_transmit_ens33", "cpu_busy"]
    }

    rif = ResidualIsolationForest(
        ind_cols=IND_COLS,
        env_cols=ENV_COLS,
        contamination=0.009,
        random_state=42,
        residual_strategy="oob",
        bayes_search=True,
        iso_params={"max_features": 1},
    )

    rif.fit(X_scaled_df)

    y_pred_rif = np.where(rif.predict(X_scaled_df) == -1, 1, 0)
    df["anomaly_rif"] = y_pred_rif

    # rif.get_feature_mapping()
    #
    # from sklearn.model_selection import cross_val_score
    #
    # scores = cross_val_score(
    #     ResidualIsolationForest(ind_cols=IND_COLS, env_cols=ENV_COLS),
    #     X, y,
    #     cv=5,
    #     scoring='accuracy'
    # )  # quesot non funge
    #
    #
    #
    #
    # from sklearn.model_selection import GridSearchCV
    #
    # param_grid = {
    #     'contamination': [0.05, 0.10, 0.15],
    #     'bayes_search': [True, False],
    #     'residual_strategy': ['oob', 'kfold', None]
    # }
    #
    # grid_search = GridSearchCV(
    #     ResidualIsolationForest(ind_cols=IND_COLS, env_cols=ENV_COLS),
    #     param_grid,
    #     cv=3,
    #     scoring='f1'
    # )
    # grid_search.fit(X, y)























    acc_rif = accuracy_score(y, y_pred_rif)
    rec_rif = recall_score(y, y_pred_rif)
    cm_rif = confusion_matrix(y, y_pred_rif)
    print("=== Residual Isolation Forest ===")
    print(f"Accuracy : {acc_rif:.3f}")
    print(f"Recall   : {rec_rif:.3f}")
    print("Confusion matrix:\n", cm_rif)


    X.drop(columns=["anomaly_type"], inplace=True)
    # Isolation Forest vanilla
    iso = IsolationForest(
        contamination=0.009,
        random_state=42,
        max_features=1
    ).fit(X_scaled_df)
    iso_pred = np.where(iso.predict(X_scaled_df) == -1, 1, 0)
    df["anomaly_iso"] = iso_pred
    acc_iso = accuracy_score(y, iso_pred)
    rec_iso = recall_score(y, iso_pred)
    cm_iso = confusion_matrix(y, iso_pred)
    print("\n=== Isolation Forest (vanilla) ===")
    print(f"Accuracy : {acc_iso:.3f}")
    print(f"Recall   : {rec_iso:.3f}")
    print("Confusion matrix:\n", cm_iso)

    from sklearn.cluster import DBSCAN

    X_scaled_df.drop(columns=["hour"], inplace=True)

    eps_value = np.mean(np.std(X_scaled_df.values, axis=0)) / 2 * 3.5

    dbscan = DBSCAN(eps=eps_value, min_samples=5)

    db_labels = dbscan.fit_predict(X_scaled_df)

    dbscan_pred = np.where(db_labels == -1, 1, 0)
    df["anomaly_dbscan"] = dbscan_pred


    acc_dbscan = accuracy_score(y, dbscan_pred)
    rec_dbscan = recall_score(y, dbscan_pred)
    cm_dbscan = confusion_matrix(y, dbscan_pred)

    print("\n=== DBSCAN ===")
    print(f"Accuracy : {acc_dbscan:.3f}")
    print(f"Recall   : {rec_dbscan:.3f}")
    print("Confusion matrix:\n", cm_dbscan)

    #df.to_excel(r"C:\Users\loverdegiulio\PycharmProjects\RIF_estimator_scikit\test_eif\datas\results2.xlsx", index=False)


