import numpy as np
from sklearn.utils.validation import check_is_fitted
from skopt.space import Integer
from rif_estimator.residual_gen import ResidualGenerator
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
            residual_strategy: str = "oob",
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

        res_train = self.generator.fit_transform(X)

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




