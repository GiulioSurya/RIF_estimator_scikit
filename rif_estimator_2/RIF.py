import numpy as np
from sklearn.utils.validation import check_is_fitted
from skopt.space import Integer
from rif_estimator_2.residual_gen import ResidualGenerator
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.ensemble import IsolationForest
from typing import Sequence, Dict, Optional


class ResidualIsolationForest(BaseEstimator, OutlierMixin):
    """Contextual anomaly detector (Residual + Isolation Forest)."""

    def __init__(
            self,
            ind_cols: Sequence[str],
            env_cols: Sequence[str],
            *,
            contamination: float = 0.10,
            residual_strategy: str = None,
            bayes_search: bool = False,
            bayes_iter: int = 3,
            bayes_cv: int = 3,
            rf_search_space: Optional[Dict[str, Integer]] = None,
            rf_params: Optional[Dict] = None,
            iso_params: Optional[Dict] = None,
            random_state: Optional[int] = None,
    ):
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

        res_train = self.generator.fit_transform(X)  # qua c'è il problema in caso di data leakage

        self.if_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            **self.iso_params,
        ).fit(res_train)
        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "if_")
        res = self.generator.transform(X)
        return self.if_.predict(res)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Compute the anomaly score of each sample.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples.

        Returns
        -------
        np.ndarray
            The anomaly score of the input samples.
            The lower, the more abnormal. Negative scores represent outliers,
            positive scores represent inliers.
        """
        check_is_fitted(self, "if_")
        res = self.generator.transform(X)
        return self.if_.decision_function(res)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

    df = pd.read_csv(r"C:\Users\loverdegiulio\PycharmProjects\tesi\datas\synthetic_data_testX.csv")
    # df = pd.read_csv(r"C:\Users\loverdegiulio\PycharmProjects\tesi\datas\synthetic_data_moderate.csv")

    ENV_COLS = ["env_X0", "env_X1", "env_X2", "env_X3", "env_X4", "env_X5"]
    IND_COLS = ["ind_Y0", "ind_Y1", "ind_Y2"]

    y = df["is_anomaly"].to_numpy()
    X = df.drop(columns=["is_anomaly", "is_outlier"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
    )

    rif = ResidualIsolationForest(
        ind_cols=IND_COLS,
        env_cols=ENV_COLS,
        contamination=0.20,
        random_state=42,
        residual_strategy="oob",
        bayes_search=True,
        iso_params={"max_features": 1}
    )
    rif.fit(X_train)

    scores = rif.decision_function(X_train)

    y_pred_test = np.where(rif.predict(X_test) == -1, 1, 0)

    acc_rif = accuracy_score(y_test, y_pred_test)
    rec_rif = recall_score(y_test, y_pred_test)
    cm_rif = confusion_matrix(y_test, y_pred_test)

    print(f"Accuracy : {acc_rif:.3f}")
    print(f"Recall   : {rec_rif:.3f}")
    print("Confusion matrix:\n", cm_rif)

    iso = IsolationForest(contamination=0.20, random_state=42, max_features=1).fit(X_train)
    iso_pred_test = np.where(iso.predict(X_train) == -1, 1, 0)

    acc_iso = accuracy_score(y_train, iso_pred_test)
    rec_iso = recall_score(y_train, iso_pred_test)
    cm_iso = confusion_matrix(y_train, iso_pred_test)

    print(f"Accuracy : {acc_iso:.3f}")
    print(f"Recall   : {rec_iso:.3f}")
    print("Confusion matrix:\n", cm_iso)