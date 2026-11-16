# ML Foundation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline%20Compliant-orange?logo=scikit-learn&logoColor=white)
![Testing](https://img.shields.io/badge/Tests-Zero%20Leakage%20Verified-brightgreen)

A production-oriented machine learning engineering repository focused on **leak-free feature engineering**, **custom Scikit-Learn estimators**, **statistical distribution diagnostics**, and **dimensionality reduction**.

---

## Core Engineering Standards

* **Strict Zero Data Leakage:** Every custom transformer computes parameters ($\mu$, $\sigma$, quartiles, empirical quantiles, and sampling pools) exclusively inside `.fit()` on training folds before transforming validation, holdout, or inference payloads.
* **Native Scikit-Learn Compatibility:** All estimators inherit from `BaseEstimator` and `TransformerMixin`, supporting `ColumnTransformer`, `Pipeline`, `sklearn.base.clone()`, and `GridSearchCV`.
* **Separation of Concerns:** Reusable library estimators live in `modules/`, exploratory experiments in `notebooks/`, automated verification in `tests/`, and generated diagnostics in `visualizations/`.

---

## Repository Structure

```text
ML Foundation/
├── artifacts/                                      # Serialized models and intermediate pipeline outputs
├── data/                                           # Local datasets and benchmark inputs
├── modules/
│   └── feature_engineering/
│       ├── __init__.py                             # Package exports for custom estimators
│       ├── transformers.py                         # Production Scikit-Learn custom transformer library
│       ├── master_pipeline_benchmark.py            # Multi-stage ablation benchmark (pre-PCA capstone)
│       ├── outlier_detection_and_treatment.py      # Linear vs. tree sensitivity & IQR capping comparison
│       ├── outlier_zscore.py                       # 3-Sigma empirical rule for Gaussian distributions
│       ├── outlier_iqr.py                          # Tukey's IQR proximity fences for skewed features
│       ├── outlier_percentile_winsorization.py     # Empirical percentile trimming & Winsorization
│       ├── feature_construction_and_splitting.py   # Domain synthesis & tidy atomic string splitting
│       └── curse_of_dimensionality_simulation.py   # Distance concentration & Hughes phenomenon simulation
├── notebooks/                                      # Exploratory data analysis & prototyping notebooks
├── tests/
│   ├── __init__.py
│   └── test_transformers.py                        # Unit tests for bound immutability, NaN handling & GridSearchCV
├── visualizations/                                 # KDE, boxplot, heatmap, and ablation benchmark figures
├── requirements.txt
└── README.md
```

---

## Custom Transformer Library (`modules/feature_engineering/transformers.py`)

| Estimator | Target Distribution | Mechanism | Key Hyperparameters |
| :--- | :--- | :--- | :--- |
| **`RandomSampleImputer`** | Missing at Random (MAR/MCAR) | Draws random replacements with replacement from observed training values to preserve variance. | `random_state=42` |
| **`ZScoreCapper`** | Symmetric / Gaussian | Clamps values outside $[\mu - k\sigma,\ \mu + k\sigma]$. | `factor=3.0` |
| **`IQRCapper`** | Skewed / Asymmetric | Clamps values outside Tukey's fences $[Q_1 - k\cdot IQR,\ Q_3 + k\cdot IQR]$. | `factor=1.5` |
| **`Winsorizer`** | Empirical / Arbitrary | Clamps tail observations to learned percentile boundaries $[P_{\alpha},\ P_{100-\alpha}]$. | `lower_quantile=0.01`, `upper_quantile=0.99` |

### Pipeline Integration Example

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from modules.feature_engineering.transformers import ZScoreCapper, IQRCapper

preprocessor = ColumnTransformer([
    ("gaussian_features", ZScoreCapper(factor=3.0), ["cgpa"]),
    ("skewed_features", IQRCapper(factor=1.5), ["fare"]),
])

model_pipeline = Pipeline([
    ("outliers", preprocessor),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression()),
])
```

---

## Quickstart & Verification

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the end-to-end pre-PCA feature engineering ablation benchmark
python modules/feature_engineering/master_pipeline_benchmark.py

# 3. Execute the automated data-leakage and estimator unit test suite
pytest -v
```

---

## Curriculum & Engineering Roadmap

* [x] **Missing Data Handling:** Univariate imputation, Missing Indicators, Random Sample Imputation, and MICE (`IterativeImputer`)
* [x] **Feature Encoding & Discretization:** One-Hot, Ordinal, Frequency/Target encoding, and Quantile/K-Means binning
* [x] **Outlier Detection & Treatment:** Z-Score ($3\sigma$), Tukey's IQR Proximity Rule, and Percentile Winsorization
* [x] **Feature Construction & Splitting:** Domain feature synthesis and Tidy Data atomic string extraction
* [x] **High-Dimensional Diagnostics & Testing:** Curse of Dimensionality simulation and automated `pytest` verification
* [ ] **Feature Extraction:** Principal Component Analysis (PCA) — mathematical foundations, eigendecomposition, and high-dimensional pipelines