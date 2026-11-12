import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error

print("=" * 65)
print("FEATURE ENGINEERING: PERCENTILE METHOD & WINSORIZATION")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC WEIGHT-HEIGHT DATASET (10,000 RECORDS)
# -----------------------------------------------------------
np.random.seed(42)
n_samples = 10000

# Height in inches (Mean ~ 66.37, Std ~ 3.85) with extreme tall/short anomalies
height = np.random.normal(loc=66.37, scale=3.85, size=n_samples)
height[:25] = np.random.uniform(77.5, 84.0, size=25)    # Extreme upper anomalies
height[25:50] = np.random.uniform(48.0, 54.0, size=25)  # Extreme lower anomalies

# Weight in pounds correlated with Height
weight = (height * 5.2) - 185 + np.random.normal(loc=0, scale=14, size=n_samples)

df = pd.DataFrame({
    "height_in": height,
    "weight_lbs": weight
})

print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns.")
print("\nRaw Height Summary Statistics:")
print(df["height_in"].describe().round(2))

# Leakage-Free Train-Test Split
X = df[["height_in"]]
y = df["weight_lbs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 2. PERCENTILE BOUNDARY AUDIT (STRICTLY ON X_TRAIN)
# -----------------------------------------------------------
print("\n--- 2. Computing Symmetric Percentile Thresholds (X_train) ---")

lower_limit = X_train["height_in"].quantile(0.01)
upper_limit = X_train["height_in"].quantile(0.99)

print(f"1st Percentile (Lower Limit):  {lower_limit:.3f} in")
print(f"99th Percentile (Upper Limit): {upper_limit:.3f} in")

outlier_mask = (X_train["height_in"] < lower_limit) | (X_train["height_in"] > upper_limit)
print(f"Outliers Flagged in X_train:   {outlier_mask.sum()} rows ({outlier_mask.mean():.2%})")

# -----------------------------------------------------------
# 3. TRIMMING VS. WINSORIZATION DIAGNOSTIC COMPARISON
# -----------------------------------------------------------
os.makedirs("visualizations", exist_ok=True)

# Approach A: Trimming (Dropping rows outside [P_01, P_99])
X_train_trimmed = X_train[~outlier_mask].copy()

# Approach B: Winsorization (Clamping values to [P_01, P_99])
X_train_winsorized = X_train.copy()
X_train_winsorized["height_in"] = X_train_winsorized["height_in"].clip(
    lower=lower_limit, upper=upper_limit
)

print(f"\nRow Count Comparison -> Original: {len(X_train)} | Trimmed: {len(X_train_trimmed)} | Winsorized: {len(X_train_winsorized)}")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Plot 1: Trimming KDE
X_train["height_in"].plot(kind="kde", ax=axes[0, 0], color="crimson", lw=2, label="Original")
X_train_trimmed["height_in"].plot(kind="kde", ax=axes[0, 0], color="navy", lw=2, linestyle="--", label="Trimmed [P1, P99]")
axes[0, 0].axvline(lower_limit, color="black", linestyle=":", label=f"P1 ({lower_limit:.1f})")
axes[0, 0].axvline(upper_limit, color="black", linestyle=":", label=f"P99 ({upper_limit:.1f})")
axes[0, 0].set_title("Percentile Trimming: KDE Comparison")
axes[0, 0].legend()

# Plot 2: Trimming Boxplot
axes[0, 1].boxplot(
    [X_train["height_in"], X_train_trimmed["height_in"]],
    tick_labels=["Original", "Post-Trimming"],
    patch_artist=True,
    boxprops=dict(facecolor="lightsteelblue")
)
axes[0, 1].set_title("Percentile Trimming: Boxplot")

# Plot 3: Winsorization KDE (Notice slight density bumps at the tail caps)
X_train["height_in"].plot(kind="kde", ax=axes[1, 0], color="crimson", lw=2, label="Original")
X_train_winsorized["height_in"].plot(kind="kde", ax=axes[1, 0], color="forestgreen", lw=2, linestyle="--", label="Winsorized [P1, P99]")
axes[1, 0].axvline(lower_limit, color="black", linestyle=":")
axes[1, 0].axvline(upper_limit, color="black", linestyle=":")
axes[1, 0].set_title("Winsorization (Capping): KDE (Note Tail Cap Bumps)")
axes[1, 0].legend()

# Plot 4: Winsorization Boxplot
axes[1, 1].boxplot(
    [X_train["height_in"], X_train_winsorized["height_in"]],
    tick_labels=["Original", "Post-Winsorization"],
    patch_artist=True,
    boxprops=dict(facecolor="lightgreen")
)
axes[1, 1].set_title("Winsorization (Capping): Boxplot")

plt.tight_layout()
plt.savefig("visualizations/percentile_winsorization_audit.png")
plt.close()
print("Saved diagnostic plots -> 'visualizations/percentile_winsorization_audit.png'")

# -----------------------------------------------------------
# 4. PRODUCTION SCIKIT-LEARN TRANSFORMER: WINSORIZER
# -----------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Caps numerical features at specified lower and upper quantiles.
    Learns quantile cutoffs strictly during .fit() to prevent data leakage.
    """
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.bounds_ = {}
        for col in X_df.columns:
            self.bounds_[col] = {
                "lower": X_df[col].quantile(self.lower_quantile),
                "upper": X_df[col].quantile(self.upper_quantile)
            }
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, bound in self.bounds_.items():
            X_df[col] = X_df[col].clip(lower=bound["lower"], upper=bound["upper"])
        return X_df.values

# -----------------------------------------------------------
# 5. AUTOMATED QUANTILE TUNING VIA GRIDSEARCHCV
# -----------------------------------------------------------
print("\n--- 5. Tuning Optimal Winsorization Quantiles via GridSearchCV ---")

pipeline = Pipeline([
    ("winsorizer", Winsorizer()),
    ("scaler", StandardScaler()),
    ("regressor", Ridge())
])

param_grid = [
    {"winsorizer__lower_quantile": [0.005], "winsorizer__upper_quantile": [0.995]},
    {"winsorizer__lower_quantile": [0.01],  "winsorizer__upper_quantile": [0.99]},
    {"winsorizer__lower_quantile": [0.025], "winsorizer__upper_quantile": [0.975]},
    {"winsorizer__lower_quantile": [0.0],   "winsorizer__upper_quantile": [1.0]}  # No capping baseline
]

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print(f"Best Quantile Cutoffs: {grid.best_params_}")
print(f"Best 5-Fold CV RMSE:   {-grid.best_score_:.3f} lbs")

test_preds = grid.predict(X_test)
test_rmse = root_mean_squared_error(y_test, test_preds)
print(f"Holdout Test RMSE:     {test_rmse:.3f} lbs")

# -----------------------------------------------------------
# 6. INFERENCE ON EXTREME UNSEEN PAYLOAD
# -----------------------------------------------------------
print("\n--- 6. Direct Inference on Extreme Unprocessed Input ---")

unseen_person = pd.DataFrame([{"height_in": 86.0}])  # Extreme 7'2" outlier -> clamped to P99
predicted_weight = grid.predict(unseen_person)[0]

print(f"Incoming Raw Input: {unseen_person.to_dict(orient='records')[0]}")
print(f"Predicted Weight:   {predicted_weight:.2f} lbs")