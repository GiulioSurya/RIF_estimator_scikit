# Residual Isolation Forest (RIF)

**Residual Isolation Forest (RIF)** is a scikit‑learn‑compatible estimator for **contextual anomaly detection (CAD)**.
It augments classic Isolation Forest by first modelling the expected behaviour via contextual regression, then detecting anomalies on the *residuals* of that regression.

This repository provides two building blocks:

| Module                    | Role                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `ResidualGenerator`       | Fits a Random Forest *env → ind* to obtain **leakage‑free residuals** using one of three strategies (None, OOB, K‑fold). |
| `ResidualIsolationForest` | Applies Isolation Forest on those residuals to flag outliers.                                                            |

---

## 🔍 Motivation

Raw behavioural signals often drift just because the environment changes. Treating every deviation as an anomaly triggers many false alarms.
RIF embraces **contextual anomaly detection**:

1. **Context learning** – regress behavioural variables *Y* on environmental variables *X*.
2. **Residual extraction** – compute `R = Y − Ŷ` as behaviour unexplained by the context.
3. **Anomaly detection** – run Isolation Forest on *R*.

This decouples legitimate contextual variability from true anomalies.

---

## 🧠 How it works

### End‑to‑end pipeline

```text
fit(X_train)
└─▶ ResidualGenerator.fit_transform(X_train)
     ├─ Fit Random Forest env→ind (one RF per Y)
     └─ Compute residuals_train      ← strategy‑dependent
└─▶ IsolationForest.fit(residuals_train)

predict(X_test)
└─▶ ResidualGenerator.transform(X_test)
     └─ Compute residuals_test       ← always out‑of‑sample
└─▶ IsolationForest.predict(residuals_test)
```

### Residual strategies

| Strategy               | How residuals\_train are computed                                                                                   | Pros                                                   | Cons                                                                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **`"none"`** (leakage) | RF predicts the very same samples it was fitted on. Residuals for normal points collapse around 0.                  | Fast; best recall if anomalies are strongly separated. | Heavy **data‑leakage**: threshold learned on an artefact; may explode false positives or false negatives when signal/noise ratio changes. |
| **`"oob"`**            | Uses the *out‑of‑bag* predictions of each tree. Every training row is predicted by trees that have **not** seen it. | Leakage‑free, quick, no extra splits.                  | Can contain `NaN` when RF has few trees; residuals noisier ⇒ recall drops unless threshold is tuned.                                      |
| **`"kfold"`**          | K‑fold cross‑validation: each fold is predicted by a model trained on the others.                                   | Fully leakage‑free, deterministic, no `NaN`.           | Slow (trains K RFs), larger memory footprint.                                                                                             |

> **Where it acts**: the strategy only affects **`ResidualGenerator.fit_transform()`** – i.e. the residuals used to *train* the Isolation Forest.
> At prediction time (`transform()`), residuals are always computed *out‑of‑sample* because the test set is new.

### Choosing a strategy

| If you want…                                              | Pick    | Tune                                                                                  |
| --------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------- |
| Maximum speed & highest recall **and** you accept leakage | `none`  | Use validation PR‑curve to set a custom threshold instead of default `contamination`. |
| Balanced trade‑off, leakage‑free                          | `oob`   | Increase `n_estimators` (≥200) to reduce `NaN`; maybe scale residuals.                |
| Production robustness & reproducibility                   | `kfold` | Choose `k`; expect `k×` training time.                                                |

---

## 📦 Installation

```bash
pip install git+https://github.com/GiulioSurya/RIF_estimator_scikit.git
```

---

## ⚙️ Quick start

```python
from rif_estimator import ResidualIsolationForest

IND_COLS = ["ind_Y0", "ind_Y1", "ind_Y2"]
ENV_COLS = ["env_X0", "env_X1", "env_X2", "env_X3", "env_X4", "env_X5"]

rif = ResidualIsolationForest(
    ind_cols=IND_COLS,
    env_cols=ENV_COLS,
    contamination=0.20,
    residual_strategy="oob",   # "none" or "kfold" also allowed
    bayes_search=True,          # Bayesian tuning of each RF
    iso_params={"max_features": 1},
)

rif.fit(X_train)
labels = rif.predict(X_test)          # -1 = outlier, 1 = inlier
scores = rif.decision_function(X_test)
```

---

## 🛠️ Feature highlights

* **Scikit‑learn API** (`fit`, `predict`, `decision_function`)
* **Leakage‑free residuals** via OOB or K‑fold
* **Bayesian hyper‑parameter optimisation** (skopt `BayesSearchCV`)
* **Multi‑output ready** (multiple behavioural `Y`)
* **Hash‑based cache** – avoids recomputing residuals when the same DataFrame is passed to `transform()`.

---
