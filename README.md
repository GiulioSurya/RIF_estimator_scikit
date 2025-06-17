# Residual Isolation Forest (RIF) 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/sklearn-compatible-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A scikit-learn compatible estimator for contextual anomaly detection.**

Residual Isolation Forest (RIF) is a powerful anomaly detection algorithm that combines the best of both worlds: **contextual understanding** and **unsupervised learning**. Instead of treating all deviations as anomalies, RIF first models what's *normal* given the context, then detects anomalies in the unexplained residuals.

##  Why RIF?

Traditional anomaly detection methods often produce false alarms because they don't consider *context*. **Contextual anomalies** are data points that appear normal in general but are odd in a specific situation. For example:

- **🏢 Server monitoring**: High CPU usage at 3 AM might be suspicious, but during business hours it's normal
- **🌡️ Environmental sensors**: A 30°C temperature reading is normal in summer but anomalous in winter


**RIF addresses this by:**
1. **Learning context-behavior relationships** using Random Forest regression
2. **Computing residuals** that represent behavior unexplained by context
3. **Detecting anomalies** in the residual space using Isolation Forest

---

##  How It Works

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Data      │───▶│ Residual         │───▶│   Isolation     │
│ (Context + Behavior)│    │ Generation       │    │   Forest        │
│                 │    │ (Random Forest)  │    │ (Anomaly Detection)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                        │
        │                        │                        │
    Environmental            Residuals =                Labels
    + Behavioral           Actual - Predicted         (-1: anomaly
     Variables               Behavior                  +1: normal)
```

### The Two-Stage Process

**Stage 1: Context Modeling** 
- Fits Random Forest models to predict behavioral variables from environmental features
- Learns patterns like "CPU usage should be high when network traffic is high"
- Supports different environmental predictors for each behavioral variable

**Stage 2: Anomaly Detection**   
- Computes residuals (actual - predicted behavior)
- Applies Isolation Forest to identify outliers in residual space
- Residuals represent "behavior that can't be explained by context"

---

##  Quick Start

### Installation

```bash
pip install git+https://github.com/GiulioSurya/RIF_estimator_scikit.git
```

### Basic Usage

```python
import pandas as pd
from rif_estimator import ResidualIsolationForest

# Your data with environmental and behavioral columns
df = pd.read_csv('your_data.csv')

# Define which columns represent context vs behavior
ENV_COLS = ['time_of_day', 'day_of_week', 'temperature', 'load']
IND_COLS = ['cpu_usage', 'memory_usage', 'response_time']

# Create and train the model
rif = ResidualIsolationForest(
    ind_cols=IND_COLS,           # What to monitor (behavioral)
    env_cols=ENV_COLS,           # What explains it (environmental)  
    contamination=0.1,           # Expected anomaly rate
    residual_strategy='oob',     # Leakage-free residuals
    bayes_search=True            # Auto-tune Random Forest
)

# Fit on training data
rif.fit(X_train)

# Detect anomalies
predictions = rif.predict(X_test)     # -1 = anomaly, +1 = normal
anomaly_scores = rif.decision_function(X_test)  # Lower = more anomalous
```

### Advanced Usage

```python
# Different environmental predictors for each behavioral variable
feature_mapping = {
    'cpu_usage': ['network_traffic', 'running_processes', 'memory_usage'],
    'disk_io': ['file_operations', 'cpu_usage'],
    'network_latency': ['network_traffic', 'server_load']
}

rif = ResidualIsolationForest(
    ind_cols=feature_mapping,    # Dict mapping targets to predictors
    contamination=0.05,
    residual_strategy='kfold',   # Most robust strategy
    bayes_search=True,
    bayes_iter=10,              # More thorough hyperparameter search
    iso_params={'max_features': 1, 'max_samples': 0.8}
)
```

---

## ️ Key Parameters

### Core Configuration

| Parameter | Description | Default | Recommendations |
|-----------|-------------|---------|-----------------|
| `ind_cols` | **Behavioral variables** to monitor for anomalies | Required | Choose metrics that represent system behavior |
| `env_cols` | **Environmental variables** that explain behavior | Required | Select features that logically influence behavior |
| `contamination` | Expected proportion of anomalies | `0.10` | Tune based on domain knowledge (0.01-0.2) |

### Residual Strategies 

The **most important parameter** for avoiding data leakage:

| Strategy | Description | Speed | Accuracy | Use When |
|----------|-------------|--------|----------|-----------|
| `'none'` | ⚠️ **Leaky**: Direct predictions | 🚀 Fastest | ⚡ High recall | Prototyping only |
| `'oob'` | **Out-of-bag** predictions | 🏃 Fast | 📊 Balanced | Production default |  
| `'kfold'` | **Cross-validation** predictions | 🐌 Slower | 🎯 Most robust | Critical applications |

### Performance Tuning

| Parameter | Description | Impact |
|-----------|-------------|---------|
| `bayes_search=True` | Auto-tune Random Forest hyperparameters | Better accuracy, slower training |
| `bayes_iter=10` | Hyperparameter search iterations | More = better tuning |
| `iso_params={'max_features': 1}` | Isolation Forest parameters | Affects anomaly detection sensitivity |

---

##  Real-World Examples

### Server Monitoring

```python
# Monitor server performance with context awareness
server_rif = ResidualIsolationForest(
    ind_cols=['cpu_usage', 'memory_usage', 'disk_io'],
    env_cols=['hour', 'day_of_week', 'active_users', 'network_traffic'],
    contamination=0.02,  # Expect 2% anomalies
    residual_strategy='oob'
)

# Detects: CPU spikes not explained by user load or time patterns
```

###  Environmental Monitoring

```python
# Different predictors for different sensors
sensor_mapping = {
    'temperature': ['season', 'time_of_day', 'humidity', 'solar_radiation'],
    'humidity': ['temperature', 'pressure', 'wind_speed'],
    'air_quality': ['wind_speed', 'temperature', 'traffic_density']
}

env_rif = ResidualIsolationForest(
    ind_cols=sensor_mapping,
    contamination=0.05,
    residual_strategy='kfold',  # Maximum robustness
    bayes_search=True
)
```

---

##  Understanding Your Results

### Basic Predictions and Scoring

```python
# Get binary predictions
predictions = rif.predict(X_test)  # -1 for anomalies, +1 for normal
anomaly_mask = predictions == -1
print(f"Found {np.sum(anomaly_mask)} anomalies out of {len(X_test)} samples")

# Get continuous anomaly scores 
scores = rif.decision_function(X_test)  # Lower = more anomalous
most_anomalous_idx = np.argsort(scores)[:10]
print(f"Most anomalous samples: {most_anomalous_idx}")
```

### Advanced Scoring Methods

RIF provides multiple ways to assess anomaly likelihood:

```python
# Raw anomaly scores (before applying contamination threshold)
raw_scores = rif.score_samples(X_test)

# Decision function scores (relative to threshold)
decision_scores = rif.decision_function(X_test)

# Access the threshold used to separate normal from anomalous
threshold = rif.offset_
print(f"Anomaly threshold: {threshold:.3f}")

# Verify the relationship: decision_function = score_samples - offset
assert np.allclose(decision_scores, raw_scores - threshold)
```

### Score Interpretation

```python
# Understanding the different scoring methods:

# 1. score_samples(): Raw isolation scores
raw_scores = rif.score_samples(X_test)
print("Raw scores - higher values = more normal")
print(f"Range: [{raw_scores.min():.3f}, {raw_scores.max():.3f}]")

# 2. decision_function(): Threshold-adjusted scores  
decision_scores = rif.decision_function(X_test)
print("Decision scores - negative values = anomalies")
print(f"Range: [{decision_scores.min():.3f}, {decision_scores.max():.3f}]")

# 3. offset_: The threshold that separates normal from anomalous
print(f"Threshold (offset): {rif.offset_:.3f}")
print(f"Samples with decision_function < 0: {np.sum(decision_scores < 0)} (predicted anomalies)")
```

### Feature Mapping

```python
# See which environmental features explain which behaviors
feature_mapping = rif.get_feature_mapping()
for target, predictors in feature_mapping.items():
    print(f"{target} is predicted by: {', '.join(predictors)}")
```

---

##  Scikit-learn Compatibility

RIF is fully compatible with scikit-learn's ecosystem:

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Cross-validate anomaly detection performance
# Note: Convert labels to scikit-learn format (1=normal, -1=anomaly)
y_sklearn = np.where(y_true == 1, -1, 1)  # Convert your labels

rif = ResidualIsolationForest(
    ind_cols=['cpu_usage', 'memory_usage'],
    env_cols=['hour', 'network_traffic'],
    contamination=0.1
)

# Perform cross-validation
cv_scores = cross_val_score(
    rif, X, y_sklearn, 
    cv=5, 
    scoring='accuracy'
)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

### Grid Search Compatibility

```python
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'contamination': [0.05, 0.1, 0.15],
    'residual_strategy': ['oob', 'kfold'],
    'bayes_search': [True, False]
}

grid_search = GridSearchCV(
    ResidualIsolationForest(ind_cols=IND_COLS, env_cols=ENV_COLS),
    param_grid,
    cv=3,
    scoring='recall',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
```

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Use RIF in a scikit-learn pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rif', ResidualIsolationForest(
        ind_cols=IND_COLS,
        env_cols=ENV_COLS,
        contamination=0.1
    ))
])

# Fit and predict with the pipeline
pipeline.fit(X_train)
predictions = pipeline.predict(X_test)
```

---

## Performance Tips

### Speed Optimization

```python
# For large datasets, prioritize speed
fast_rif = ResidualIsolationForest(
    ind_cols=IND_COLS,
    env_cols=ENV_COLS,
    residual_strategy='oob',     # Faster than k-fold
    bayes_search=False,          # Skip hyperparameter tuning
    rf_params={'n_estimators': 100, 'max_depth': 10},  # Fixed params
    iso_params={'n_estimators': 50}  # Fewer trees
)
```

###  Accuracy Optimization

```python
# For critical applications, maximize accuracy
robust_rif = ResidualIsolationForest(
    ind_cols=IND_COLS,
    env_cols=ENV_COLS,
    residual_strategy='kfold',   # Most robust strategy
    bayes_search=True,
    bayes_iter=10,              # Thorough hyperparameter search
    bayes_cv=5                  # More CV for hyperparameter tuning
)
```

## ⚠️ Critical: Data Integrity Warning

> **🚨 IMPORTANT: Avoid Data Leakage When Evaluating on Training Data**

RIF uses **DataFrame fingerprinting** to detect when you're applying the model to the same data used for training. This is crucial for maintaining leakage-free residuals:

### ✅ **Safe Operations** (Preserves fingerprint)
```python
# Train the model
rif.fit(X_train)

# These operations are SAFE - fingerprint remains valid
predictions = rif.predict(X_train)      # ✅ Uses cached OOB/K-fold residuals
scores = rif.decision_function(X_train) # ✅ No data leakage
```

###  **Dangerous Operations** (Breaks fingerprint)
```python
# Train the model
rif.fit(X_train)

# These operations BREAK the fingerprint and cause data leakage:
X_train_modified = X_train.reset_index(drop=True)  # ❌ Index change
X_train_sorted = X_train.sort_values('column')     # ❌ Row reordering  
X_train_subset = X_train[X_train['col'] > 0]       # ❌ Filtering
X_train_new_col = X_train.assign(new_col=1)        # ❌ Column addition

# Using modified data causes LEAKAGE:
predictions = rif.predict(X_train_modified)  # ❌ Uses direct predictions!
```

### 🛡 **Why This Matters**
- **Fingerprint match**: Uses proper OOB/K-fold residuals → **No leakage**
- **Fingerprint broken**: Falls back to direct predictions → **Data leakage**
- **Result**: Invalid evaluation metrics and false confidence in model performance

###  **Best Practices**
```python
# ✅ CORRECT: Keep training data unchanged
X_train_original = X_train.copy()  # Save original
rif.fit(X_train_original)

# Evaluate on original data
train_predictions = rif.predict(X_train_original)  # Safe
test_predictions = rif.predict(X_test)             # Always safe

# ✅ ALTERNATIVE: Use separate validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
rif.fit(X_train)
val_predictions = rif.predict(X_val)  # Always safe (different data)
```

##  Troubleshooting

### Common Issues and Solutions

**🚨 "OOB predictions contain NaN"**
```python
# Solution: Use more trees or switch strategy
rif = ResidualIsolationForest(
    residual_strategy='kfold',  # or increase n_estimators
    rf_params={'n_estimators': 200}
)
```

**🚨 "Poor recall on anomalies"**
```python
# Solution: Tune contamination parameter
from sklearn.metrics import precision_recall_curve

# Use validation set to find optimal threshold
scores = rif.decision_function(X_val)
precision, recall, thresholds = precision_recall_curve(y_val, -scores)
# Plot and choose threshold that balances precision/recall
```

**🚨 "Training is too slow"**
```python
# Solution: Reduce complexity
rif = ResidualIsolationForest(
    bayes_search=False,
    residual_strategy='oob',
    rf_params={'n_estimators': 50, 'max_depth': 10}
)
```

**🚨 "Too many false positives"**
```python
# Solution: Lower contamination or improve features
rif = ResidualIsolationForest(
    contamination=0.05,  # Expect fewer anomalies
    # Add more relevant environmental features
    env_cols=ENV_COLS + ['additional_context_feature']
)
```

---

## ️ Architecture Details

### Core Components

**`ResidualGenerator`** - The context modeling engine
- Fits Random Forest regressors (env → behavior)
- Computes leakage-free residuals using OOB/K-fold strategies
- Handles multiple target variables with different predictors
- Implements DataFrame fingerprinting for efficient caching

**`ResidualIsolationForest`** - The main estimator
- scikit-learn compatible interface (`fit`, `predict`, `decision_function`)
- Orchestrates residual generation and anomaly detection
- Supports Bayesian hyperparameter optimization
- Provides model introspection capabilities

### Data Flow

```
Input DataFrame
       ↓
[Split into Environmental & Behavioral columns]
       ↓
Random Forest Training (env → behavior)
       ↓
Residual Computation (actual - predicted)
       ↓  
Isolation Forest Training (on residuals)
       ↓
Anomaly Detection (in residual space)
```

---

##  Theoretical Background

RIF is based on the principle of **contextual anomaly detection**: anomalies that are aberrant data examples in a given context but otherwise normal. The algorithm addresses limitations of traditional approaches by:

1. **Modeling Expected Behavior**: Uses the relationship between environmental factors and behavioral metrics to establish baselines
2. **Isolating Unexplained Variation**: Residuals capture deviations that cannot be attributed to known contextual factors  
3. **Robust Anomaly Detection**: Applies Isolation Forest in the residual space where anomalies are more apparent

This approach is particularly effective for:
- **Time-series data** with seasonal/cyclical patterns
- **Multi-variate systems** where variables influence each other
- **Noisy environments** where context helps distinguish signal from noise

---

##  Research Foundation

The RIF algorithm draws from several key research areas:

- **Contextual Anomaly Detection**: KNN CAD and other contextual approaches that extract features to determine how much current data differs from previous patterns
- **Isolation Forest**: Efficient anomaly detection through random partitioning
- **Residual Analysis**: Classical statistical technique for removing systematic effects
- **Ensemble Methods**: Random Forest for robust regression modeling

---

##  Contributing

We welcome contributions! Please see our contributing guidelines for:
- Bug reports and feature requests
- Code contributions and improvements  
- Documentation enhancements
- New examples and use cases

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Citation

If you use RIF in your research, please cite:

```bibtex
@software{rif_estimator,
  author = {Giulio Surya Lo Verde},
  title = {Residual Isolation Forest: A scikit-learn compatible estimator for contextual anomaly detection},
  year = {2025},
  url = {https://github.com/GiulioSurya/RIF_estimator_scikit}
}
```

---

##  Support

- **Documentation**: Check the docstrings and examples above
- **Issues**: [GitHub Issues](https://github.com/GiulioSurya/RIF_estimator_scikit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GiulioSurya/RIF_estimator_scikit/discussions)

---

