import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer

print("=" * 65)
print("FEATURE ENGINEERING: POWER TRANSFORMERS (BOX-COX & YEO-JOHNSON)")
print("=" * 65)

# -----------------------------------------------------------
# 1. LOAD BENCHMARK DATASET (CONCRETE COMPRESSIVE STRENGTH)
# -----------------------------------------------------------
# Features: cement, slag, ash, water, superplastic, coarseagg, fineagg, age
# Target: strength (continuous numerical output)
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv"

try:
    df = pd.read_csv(url)
    print(f"Dataset successfully loaded from remote source ({df.shape[0]} rows, {df.shape[1]} columns).")
except Exception:
    # Synthetic fallback replicating exact concrete dataset properties
    np.random.seed(42)
    n = 1030
    df = pd.DataFrame({
        "cement": np.random.gamma(shape=5, scale=50, size=n),
        "slag": np.random.choice([0.0, 50.0, 120.0, 180.0], size=n, p=[0.45, 0.2, 0.2, 0.15]),
        "ash": np.random.choice([0.0, 30.0, 75.0, 115.0], size=n, p=[0.55, 0.15, 0.15, 0.15]),
        "water": np.random.normal(loc=180, scale=20, size=n),
        "superplastic": np.random.choice([0.0, 5.0, 10.0, 15.0], size=n, p=[0.35, 0.3, 0.2, 0.15]),
        "coarseagg": np.random.normal(loc=970, scale=75, size=n),
        "fineagg": np.random.normal(loc=770, scale=60, size=n),
        "age": np.random.exponential(scale=35, size=n) + 1,
        "strength": np.random.normal(loc=35, scale=15, size=n)
    })
    print(f"Loaded synthetic dataset benchmark ({df.shape[0]} rows).")

# Features & Target separation
X = df.drop(columns=[df.columns[-1]])
y = df[df.columns[-1]]

# Check zero values present across features
zero_counts = (X == 0).sum()
print("\nFeatures containing strictly zero values:")
print(zero_counts[zero_counts > 0])

# Leakage-Free Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 2. BASELINE BENCHMARK: RAW UNTRANSFORMED DATA
# -----------------------------------------------------------
print("\n--- 2. Baseline Model Performance (Raw Features) ---")
baseline_lr = LinearRegression()
baseline_cv_r2 = cross_val_score(baseline_lr, X_train, y_train, cv=10, scoring="r2").mean()

baseline_lr.fit(X_train, y_train)
baseline_test_r2 = baseline_lr.score(X_test, y_test)

print(f"Raw Features - 10-Fold CV Mean R²: {baseline_cv_r2:.4f}")
print(f"Raw Features - Test Set R²:        {baseline_test_r2:.4f}")

# -----------------------------------------------------------
# 3. BOX-COX TRANSFORMATION (REQUIRING x > 0)
# -----------------------------------------------------------
print("\n--- 3. Box-Cox Transformation ---")
# Box-Cox strictly requires x > 0.
# Adding a small epsilon offset (+0.00001) to allow execution on zero-containing columns
X_train_bc_input = X_train + 0.00001
X_test_bc_input = X_test + 0.00001

pt_boxcox = PowerTransformer(method="box-cox", standardize=True)
X_train_bc = pt_boxcox.fit_transform(X_train_bc_input)
X_test_bc = pt_boxcox.transform(X_test_bc_input)

# Model evaluation under Box-Cox
lr_bc = LinearRegression()
bc_cv_r2 = cross_val_score(lr_bc, X_train_bc, y_train, cv=10, scoring="r2").mean()
lr_bc.fit(X_train_bc, y_train)
bc_test_r2 = lr_bc.score(X_test_bc, y_test)

print(f"Box-Cox (+ offset) - 10-Fold CV Mean R²: {bc_cv_r2:.4f}")
print(f"Box-Cox (+ offset) - Test Set R²:        {bc_test_r2:.4f}")

# -----------------------------------------------------------
# 4. YEO-JOHNSON TRANSFORMATION (NATIVE ZERO & NEGATIVE SUPPORT)
# -----------------------------------------------------------
print("\n--- 4. Yeo-Johnson Transformation (Production Default) ---")
# Operates directly on original raw data without requiring manual offsets
pt_yeo = PowerTransformer(method="yeo-johnson", standardize=True)
X_train_yeo = pt_yeo.fit_transform(X_train)
X_test_yeo = pt_yeo.transform(X_test)

# Model evaluation under Yeo-Johnson
lr_yeo = LinearRegression()
yeo_cv_r2 = cross_val_score(lr_yeo, X_train_yeo, y_train, cv=10, scoring="r2").mean()
lr_yeo.fit(X_train_yeo, y_train)
yeo_test_r2 = lr_yeo.score(X_test_yeo, y_test)

print(f"Yeo-Johnson (Direct) - 10-Fold CV Mean R²: {yeo_cv_r2:.4f}")
print(f"Yeo-Johnson (Direct) - Test Set R²:        {yeo_test_r2:.4f}")

# -----------------------------------------------------------
# 5. INSPECTING OPTIMAL LAMBDA (λ) PARAMETERS
# -----------------------------------------------------------
print("\n--- 5. Learned Optimal Exponent Values (Lambdas) ---")
lambda_comparison = pd.DataFrame({
    "Feature": X.columns,
    "Raw Skewness": X_train.skew().values,
    "Box-Cox λ": pt_boxcox.lambdas_,
    "Yeo-Johnson λ": pt_yeo.lambdas_
})
print(lambda_comparison.to_string(index=False))

# -----------------------------------------------------------
# 6. VISUAL DISTRIBUTION COMPARISON (BEFORE VS AFTER)
# -----------------------------------------------------------
print("\n--- 6. Generating Distribution Diagnostics ---")
# Pick the most skewed feature to visualize impact
most_skewed_col = X_train.skew().idxmax()
col_idx = list(X.columns).index(most_skewed_col)

raw_feature = X_train[most_skewed_col].values
yj_feature = X_train_yeo[:, col_idx]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Raw Feature Distribution & Q-Q
axes[0, 0].set_title(f"Raw '{most_skewed_col}' Distribution (Skew: {stats.skew(raw_feature):.2f})")
pd.Series(raw_feature).plot(kind="kde", ax=axes[0, 0], color="crimson", lw=2)

stats.probplot(raw_feature, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title(f"Raw '{most_skewed_col}' Normal Q-Q Plot")

# Yeo-Johnson Distribution & Q-Q
axes[1, 0].set_title(f"Yeo-Johnson '{most_skewed_col}' (Skew: {stats.skew(yj_feature):.2f})")
pd.Series(yj_feature).plot(kind="kde", ax=axes[1, 0], color="navy", lw=2)

stats.probplot(yj_feature, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title(f"Yeo-Johnson '{most_skewed_col}' Normal Q-Q Plot")

plt.tight_layout()
plt.savefig("visualizations/power_transforms_comparison.png")
plt.close()
print("Saved comparison figure -> 'visualizations/power_transforms_comparison.png'")