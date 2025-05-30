

# Residual Isolation Forest (RIF)

**Residual Isolation Forest (RIF)** is a scikit-learn-compatible estimator for **contextual anomaly detection**. It extends the classic Isolation Forest by first learning expected behavior via contextual regression and then applying anomaly detection on the resulting residuals.

This repository contains two core components:

* `ResidualGenerator`: performs leakage-free regression to compute residuals from contextual variables.
* `ResidualIsolationForest`: detects anomalies using Isolation Forest on those residuals.

---

## 🔍 Motivation

In many real-world applications, the behavior of a system is strongly influenced by its environment. For example, the energy consumption of a device depends on workload and temperature. Directly applying anomaly detection to raw behavior may confuse legitimate contextual changes with true anomalies.

This project adopts a **contextual anomaly detection (CAD)** strategy:

1. Use a Random Forest model to estimate behavioral variables from environmental ones.
2. Compute residuals as deviations from expected behavior.
3. Apply Isolation Forest on these residuals to detect context-aware anomalies.

This decouples contextual variability from true anomalous behavior.

---

## 📦 Installation

You can install the estimator directly from GitHub using `pip`:

```bash
pip install git+https://github.com/GiulioSurya/RIF_estimator_scikit.git
```

---

## 🧠 Theory (Summary)

Let:

* **`X_env`** = contextual/environmental variables
* **`Y_ind`** = behavioral/indicator variables

We model:

```
Ŷ = f(X_env)
Residuals = Y_ind - Ŷ
```

These residuals highlight behavior that deviates from what is expected given the context. Isolation Forest is then applied to the residual space.

This methodology is inspired by prior work on **contextual anomaly detection** (Song et al., 2007; Calikus et al., 2020), and provides a robust, unsupervised alternative when ground truth labels are not available.

---

## ⚙️ Usage

### Minimal example

```python
from rif_estimator import ResidualIsolationForest

# Define which columns are contextual and which are behavioral
env_cols = ["temperature", "load"]
ind_cols = ["energy_usage"]

# Initialize the detector
rif = ResidualIsolationForest(
    env_cols=env_cols,
    ind_cols=ind_cols,
    contamination=0.05,
    residual_strategy="kfold",  # or "oob"
)

# Fit on the training data
rif.fit(data)

# Predict anomalies
anomaly_labels = rif.predict(data)  # -1 for outliers, 1 for inliers

# Get anomaly scores
scores = rif.decision_function(data)
```

---

## 🛠️ Features

* ✅ Scikit-learn compatible (`fit`, `predict`, `decision_function`)
* 🔁 Leakage-free residuals via:

  * Out-of-bag (OOB) estimation
  * K-fold cross-validation
* 🔍 Optional Bayesian optimization (`BayesSearchCV`) of regression hyperparameters
* 📈 Works with multivariate output (multiple behavioral variables)

---

## 🧪 Suggested Use Cases

* Environmental monitoring (e.g., sensor drift detection)
* Industrial equipment diagnostics
* Behavioral modeling with external covariates
* Smart grid and energy systems

---

## 📜 License

This project is open source and available under the MIT License.

---
