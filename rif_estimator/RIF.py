from typing import Sequence, Dict, Optional
import numpy as np
import pandas as pd
from skopt.space import Integer
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted
from _residual_gen import ResidualGenerator


class ResidualIsolationForest(BaseEstimator, OutlierMixin):
    """
    Contextual anomaly detector based on residuals and Isolation Forest.

    This estimator applies a regression model to predict individual (behavioral)
    variables (`ind_cols`) from contextual (environmental) variables (`env_cols`)
    using the `ResidualGenerator` module. The residuals from this regression are
    used as input for the Isolation Forest algorithm to detect anomalies in an
    unsupervised manner.

    Parameters
    ----------
    ind_cols : Sequence[str]
        Names of the columns representing individual (behavioral) features.
    env_cols : Sequence[str]
        Names of the columns representing contextual (environmental) features.
    contamination : float, default=0.10
        The expected proportion of anomalies in the dataset, used by Isolation Forest.
    residual_strategy : {'oob', 'kfold', 'None'}, optional
        Strategy used to compute residuals for training the Isolation Forest:
        - "oob": out-of-bag residuals using Random Forest
        - "kfold": residuals obtained via cross-validation
        - "None": residuals computed from a standard regression on the same dataset
          used for fitting (risk of overfitting)

        Residuals are cached and reused if the DataFrame passed to `predict` is
        the same as the one used in `fit`.
    bayes_search : bool, default=False
        If `True`, performs Bayesian hyperparameter optimization for the Random Forest.
        Recommended for non-linear relationships. For linear cases, it is advisable to
        set this to `False`.
    bayes_iter : int, default=3
        Number of iterations in the Bayesian search.
    bayes_cv : int, default=3
        Number of cross-validation folds used during Bayesian optimization.
    rf_search_space : dict[str, skopt.space.Integer], optional
        Search space for Random Forest hyperparameter optimization.
    rf_params : dict, optional
        Parameters to use for the Random Forest if Bayesian search is disabled.
    iso_params : dict, optional
        Additional parameters to pass to the Isolation Forest.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        ind_cols: Sequence[str],
        env_cols: Sequence[str],
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
        self.iso_params = iso_params or {}
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the residual generator and the Isolation Forest on the training data.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset containing both contextual and behavioral variables.
        y : Ignored
            Present only for compatibility with the scikit-learn API.

        Returns
        -------
        self : ResidualIsolationForest
            The fitted estimator.
        """

        res_train = self.generator.fit_transform(X)

        self.if_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            **self.iso_params,
        ).fit(res_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts whether each observation is an anomaly or not.

        If `X` matches the dataset used during `fit`, the cached residuals are reused.
        Otherwise, residuals are computed on the fly.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to be evaluated.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Returns -1 for anomalies and 1 for normal observations.
        """

        check_is_fitted(self, "if_")
        res = self.generator.transform(X)
        return self.if_.predict(res)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes an anomaly score for each observation.

        Higher values indicate more normal (less anomalous) observations.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to be evaluated.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Anomaly score for each observation.
        """

        check_is_fitted(self, "if_")
        res = self.generator.transform(X)
        return self.if_.decision_function(res)

# if __name__ == "__main__":
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
#
#     df = pd.read_csv(r"C:\Users\loverdegiulio\PycharmProjects\tesi\datas\synthetic_data_testX.csv")
#     # df = pd.read_csv(r"C:\Users\loverdegiulio\PycharmProjects\tesi\datas\synthetic_data_moderate.csv")
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
#     rif = ResidualIsolationForest(
#         ind_cols=IND_COLS,
#         env_cols=ENV_COLS,
#         contamination=0.20,
#         random_state=42,
#         residual_strategy="oob",
#         bayes_search=True,
#         iso_params={"max_features": 1}
#     )
#     rif.fit(X_train)
#
#     splits = zip([X_train, X_test], [y_train, y_test])
#
#     for x, y in splits:
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
# ########################
#
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
#
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

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

    env_cols = ["year", "month", "day", "latitude", "longitude"]
    ind_cols = ["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"]

    prepared_df = pd.read_csv(r"C:\Users\loverdegiulio\PycharmProjects\tesi\datas\elnino_prepared.csv")

    # Target e feature set
    y = prepared_df["is_anomaly"].to_numpy()
    X = prepared_df.drop(columns=["is_anomaly", "is_outlier"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
    )

    # Residual Isolation Forest
    rif = ResidualIsolationForest(
        ind_cols=ind_cols,
        env_cols=env_cols,
        contamination=0.10,
        random_state=42,
        residual_strategy="kfold",
        bayes_search=False,
        iso_params={"max_features": 1}
    )
    rif.fit(X_train)

    y_pred_rif = np.where(rif.predict(X_test) == -1, 1, 0)
    acc_rif = accuracy_score(y_test, y_pred_rif)
    rec_rif = recall_score(y_test, y_pred_rif)
    cm_rif = confusion_matrix(y_test, y_pred_rif)

    print("=== Residual Isolation Forest ===")
    print(f"Accuracy : {acc_rif:.3f}")
    print(f"Recall   : {rec_rif:.3f}")
    print("Confusion matrix:\n", cm_rif)

    # Isolation Forest vanilla
    iso = IsolationForest(
        contamination=0.10,
        random_state=42,
        max_features=1
    ).fit(X_train)

    iso_pred = np.where(iso.predict(X_test) == -1, 1, 0)
    acc_iso = accuracy_score(y_test, iso_pred)
    rec_iso = recall_score(y_test, iso_pred)
    cm_iso = confusion_matrix(y_test, iso_pred)

    print("\n=== Isolation Forest (vanilla) ===")
    print(f"Accuracy : {acc_iso:.3f}")
    print(f"Recall   : {rec_iso:.3f}")
    print("Confusion matrix:\n", cm_iso)