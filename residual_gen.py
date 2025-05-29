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


class ResidualGenerator(BaseEstimator, TransformerMixin):
    """
    ResidualGenerator
    -----------------
    Trasforma un insieme di variabili indicatrici (target) in una matrice
    di residui rispetto a una regressione sulle variabili ambientali (feature).
    Per ogni colonna indicatrice `y_i`, viene addestrata una Random Forest
    per stimare `ŷ_i = f(env)` e il residuo è calcolato come `r_i = y_i - ŷ_i`.

    Parametri
    ---------
    ind_cols : list of str
        Nomi delle colonne indicatrici (variabili target).
    env_cols : list of str
        Nomi delle colonne ambientali (variabili di input).
    strategy : {"oob", "kfold"}, default="oob"
        Strategia per il calcolo dei residui di training in modo leakage-free:
        - "oob": utilizza le Out-Of-Bag predictions della Random Forest. Più veloce
          e scalabile, ma può produrre `NaN` nei residui se il dataset è piccolo
          o se alcune osservazioni non sono mai fuori-bag.
        - "kfold": usa predizioni ottenute tramite cross-validation K-fold.
          È più lento ma garantisce residui definiti per ogni osservazione,
          a costo di maggiore complessità computazionale.
    kfold_splits : int, default=5
        Numero di fold per la strategia "kfold".
    bayes_search : bool, default=False
        Se True, applica un'ottimizzazione Bayesiana degli iperparametri
        della Random Forest per ogni target.
    bayes_iter : int, default=10
        Numero di iterazioni della ricerca bayesiana.
    bayes_cv : int, default=3
        Numero di fold per la cross-validation interna della BayesSearchCV.
    search_space : dict, optional
        Spazio di ricerca per l'ottimizzazione Bayesiana (usa _DEFAULT_RF_SPACE se None).
    rf_params : dict, optional
        Parametri statici da passare alla Random Forest (bypassano la ricerca Bayesiana).
    random_state : int or np.random.RandomState, optional
        Semenza per la riproducibilità.

    Attributi
    ---------
    train_residuals_ : np.ndarray, shape (n_samples, n_targets)
        Matrice dei residui leakage-free per il dataset di training.
        Può contenere valori `NaN` se la strategia "oob" non genera predizioni
        per tutte le osservazioni (caso raro ma possibile con dataset piccoli).
        Questi `NaN` vengono preservati e propagati senza generare errori.
    models_ : dict
        Modelli Random Forest addestrati per ciascuna colonna indicatrice.
    best_params_ : dict
        Parametri ottimali usati per ciascun modello (via BayesSearch o manuali).
    training_data_hash_ : str
        Hash MD5 del DataFrame utilizzato in fit() per riconoscere lo stesso
        dataset in transform().

    Note
    ----
    - La strategia "oob" è molto efficiente per dataset di grandi dimensioni, ma
      nei piccoli dataset può portare a residui `NaN` se una osservazione è inclusa
      in tutti i bootstrap sample (evento raro ma possibile).
    - La strategia "kfold" è più affidabile per dataset piccoli o quando è
      importante avere una matrice di residui completa e priva di `NaN`.
    - Il metodo transform() è leakage-free solo quando viene chiamato sullo stesso
      DataFrame utilizzato in fit() (riconosciuto tramite hash). Per dataset diversi,
      utilizza predizioni standard che non sono leakage-free.

    """

    def __init__(
            self,
            ind_cols: Sequence[str],
            env_cols: Sequence[str],
            *,
            strategy: str = "oob",
            kfold_splits: int = 5,
            bayes_search: bool = False,
            bayes_iter: int = 3,
            bayes_cv: int = 3,
            search_space: Optional[Dict[str, Integer]] = None,
            rf_params: Optional[Dict] = None,
            random_state: Optional[int] = None,
    ):
        if strategy not in {"oob", "kfold"}:
            raise ValueError("strategy must be 'oob' or 'kfold'")

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



    def _compute_dataframe_hash(self, X: pd.DataFrame) -> str:
        """Calcola un hash MD5 del DataFrame per identificarlo univocamente."""
        # Usa solo le colonne rilevanti (ind_cols + env_cols)
        relevant_cols = self.ind_cols + self.env_cols
        X_relevant = X[relevant_cols]

        # Converti in bytes per l'hash
        data_bytes = pd.util.hash_pandas_object(X_relevant, index=True).values.tobytes()
        return hashlib.md5(data_bytes).hexdigest()

    # ----------------------------------------------------------------- #
    # Fit
    # ----------------------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y=None) -> "ResidualGenerator":
        """
        Addestra le RF e calcola i residui di training in modo coerente con
        la strategia ("oob" o "kfold"). I residui vengono salvati in
        ``self.train_residuals_`` e possono essere restituiti da fit_transform.

        Salva anche un hash del DataFrame X per riconoscerlo in transform().
        """

        self.models_: Dict[str, object] = {}
        self.best_params_: Dict[str, Dict] = {}
        train_residuals: List[np.ndarray] = []

        # Calcola e salva l'hash del DataFrame di training
        self.training_data_hash_ = self._compute_dataframe_hash(X)

        X_env = X[self.env_cols]

        # Prepara lo splitter se serve
        if self.strategy == "kfold":
            cv = KFold(
                n_splits=self.kfold_splits, shuffle=True, random_state=self.random_state
            )

        for col in self.ind_cols:
            y_ind = X[col]

            # iperparametri RF
            params = (
                self._bayesian_search(X_env, y_ind)
                if self.bayes_search
                else self.rf_params
            )

            rf = RandomForestRegressor(
                    random_state=self.random_state,
                    oob_score=self.strategy == "oob",
                    bootstrap=self.strategy == "oob",
                    n_jobs=-1,
                    **params,
                )


            if self.strategy == "oob":
                rf_fit = rf.fit(X_env, y_ind)
                preds =rf_fit.oob_prediction_

            else: #kfold
                preds = cross_val_predict(
                    rf,
                    X_env,
                    y_ind,
                    cv=cv,
                    n_jobs=-1,
                )
                # Fit finale per l'uso in transform
                rf_fit = rf.fit(X_env, y_ind)

            # ---------------------------------------------------------------------------------
            # 2. Store
            # ---------------------------------------------------------------------------------
            self.models_[col] = rf_fit
            self.best_params_[col] = params
            train_residuals.append((y_ind - preds).to_numpy())


        self.train_residuals_ = np.column_stack(train_residuals).astype(float)
        return self

    # ----------------------------------------------------------------- #
    # Transform (nuovi dati o stesso dataset)
    # ----------------------------------------------------------------- #
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calcola la matrice residui per il DataFrame X.

        Comportamento:
        - Se X è lo stesso DataFrame utilizzato in fit() (riconosciuto tramite hash),
          restituisce direttamente self.train_residuals_ (leakage-free).
        - Se X è un DataFrame diverso, calcola residui standard utilizzando
          i modelli addestrati (.predict()).
        - Valori NaN nei residui vengono preservati e propagati senza errori.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame per cui calcolare i residui.

        Returns
        -------
        np.ndarray, shape (n_samples, n_targets)
            Matrice dei residui. Può contenere NaN se derivata da predizioni OOB
            incomplete.

        Note
        ----
        Questo metodo è leakage-free solo quando chiamato sullo stesso DataFrame
        utilizzato in fit(). Per dataset diversi, utilizza predizioni standard
        che possono introdurre data leakage se utilizzate impropriamente.
        """
        check_is_fitted(self, "models_")

        # Verifica se è lo stesso dataset utilizzato in fit()
        current_hash = self._compute_dataframe_hash(X)
        if current_hash == self.training_data_hash_:
            # Stesso dataset: restituisci i residui leakage-free calcolati in fit()
            return self.train_residuals_

        # Dataset diverso: calcola residui standard (non leakage-free)
        X_env = X[self.env_cols]
        res = np.column_stack(
            [
                X[col].to_numpy() - self.models_[col].predict(X_env)
                for col in self.ind_cols
            ]
        )
        return res.astype(float)

    # ----------------------------------------------------------------- #
    # Fit-Transform
    # ----------------------------------------------------------------- #
    def fit_transform(self, X: pd.DataFrame, y=None):
        """
        Wrapper: fit + restituzione dei residui di training leakage-free.

        Equivalente a fit(X).train_residuals_ ma più conciso.
        I residui restituiti possono contenere NaN se la strategia OOB
        non genera predizioni per tutte le osservazioni.
        """
        return self.fit(X).train_residuals_

    # ----------------------------------------------------------------- #
    # Bayesian search helper
    # ----------------------------------------------------------------- #
    def _bayesian_search(
            self, X_env: pd.DataFrame, y_ind: pd.Series
    ) -> Dict[str, int]:
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
            verbose=0,
        ).fit(X_env, y_ind)
        return opt.best_params_
