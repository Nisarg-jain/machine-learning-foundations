import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn marks IterativeImputer as experimental; import the flag first
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("=" * 65)
print("FEATURE ENGINEERING: MULTIVARIATE IMPUTATION (MICE / ITERATIVE)")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC INTER-CORRELATED DATASET (STARTUP FINANCIALS)
# -----------------------------------------------------------
np.random.seed(42)
n_records = 750

# Three naturally correlated business metrics
rd_spend = np.random.uniform(30000, 165000, size=n_records)
admin_spend = np.random.normal(loc=115000, scale=28000, size=n_records)
# Marketing is heavily correlated with R&D spend
marketing_spend = (rd_spend * 1.75) + np.random.normal(0, 32000, size=n_records)
marketing_spend = np.maximum(marketing_spend, 8000)

# Target: Profit is a linear combination of all three with noise
profit = (0.80 * rd_spend) + (0.12 * admin_spend) + (0.30 * marketing_spend) + np.random.normal(0, 14000, size=n_records)

X_raw = pd.DataFrame({
    "rd_spend": rd_spend,
    "admin_spend": admin_spend,
    "marketing_spend": marketing_spend
})

# Inject missing values (MAR pattern across all 3 features)
mask_rd = np.random.rand(n_records) < 0.15
mask_admin = np.random.rand(n_records) < 0.12
mask_mkt = np.random.rand(n_records) < 0.18

X_missing = X_raw.copy()
X_missing.loc[mask_rd, "rd_spend"] = np.nan
X_missing.loc[mask_admin, "admin_spend"] = np.nan
X_missing.loc[mask_mkt, "marketing_spend"] = np.nan

y = profit

print(f"Dataset: {X_missing.shape[0]} rows, {X_missing.shape[1]} features.")
print("\nMissing Values Count:")
print(X_missing.isna().sum())

# Leakage-Free Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_missing, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 2. STEP-BY-STEP CONVERGENCE TRACKING
# -----------------------------------------------------------
print("\n--- 2. Tracking MICE Numerical Convergence ---")

# Isolate the first missing cell in 'rd_spend' to track across iterations
target_row_idx = X_train[X_train["rd_spend"].isna()].index[0]
true_value = X_raw.loc[target_row_idx, "rd_spend"]

tracked_values = []
iteration_range = range(1, 11)

for iters in iteration_range:
    tracker = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=iters,
        random_state=42
    )
    X_imputed = tracker.fit_transform(X_train)
    row_pos = X_train.index.get_loc(target_row_idx)
    imputed_val = X_imputed[row_pos, 0]  # rd_spend is at index 0
    tracked_values.append(imputed_val)

print(f"Tracking Row #{target_row_idx} (Ground Truth R&D Spend: ${true_value:,.2f}):")
for i, val in enumerate(tracked_values, 1):
    diff_str = f" | Shift: ${abs(val - tracked_values[i-2]):,.2f}" if i > 1 else ""
    print(f"  Iteration {i:2d}: ${val:,.2f}{diff_str}")

# -----------------------------------------------------------
# 3. PERFORMANCE BENCHMARK (5-FOLD CV R² SCORE)
# -----------------------------------------------------------
print("\n--- 3. Performance Benchmark across Imputation Strategies ---")

imputer_strategies = {
    "SimpleImputer (Median)": SimpleImputer(strategy="median"),
    "KNNImputer (k=5)": KNNImputer(n_neighbors=5, weights="distance"),
    "IterativeImputer (MICE)": IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
}

for name, imp in imputer_strategies.items():
    pipe = Pipeline([
        ("imputer", imp),
        ("scaler", StandardScaler()),
        ("regressor", Ridge())
    ])
    
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
    pipe.fit(X_train, y_train)
    test_r2 = pipe.score(X_test, y_test)
    
    print(f"{name:<25} | 5-Fold CV R²: {cv_scores.mean():.4f} | Test R²: {test_r2:.4f}")

# -----------------------------------------------------------
# 4. DIAGNOSTIC VISUALIZATION: CONVERGENCE TRAJECTORY
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.5))

ax.plot(list(iteration_range), tracked_values, marker="o", color="navy", lw=2, label="MICE Imputed Value")
ax.axhline(true_value, color="crimson", linestyle="--", lw=1.8, label=f"Ground Truth (${true_value:,.0f})")

ax.set_title("MICE Convergence Trajectory: R&D Spend Stabilization")
ax.set_xlabel("Iteration Count")
ax.set_ylabel("Imputed Dollar Value ($)")
ax.set_xticks(list(iteration_range))
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("visualizations/mice_convergence_audit.png")
plt.close()
print("\nSaved diagnostic plot -> 'visualizations/mice_convergence_audit.png'")

# -----------------------------------------------------------
# 5. INFERENCE ON INCOMING PRODUCTION RECORD
# -----------------------------------------------------------
print("\n--- 5. Direct Inference on Unprocessed User Record ---")

production_pipeline = Pipeline([
    ("imputer", IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)),
    ("scaler", StandardScaler()),
    ("regressor", Ridge())
])

production_pipeline.fit(X_train, y_train)

# Incoming payload with missing admin_spend
raw_payload = pd.DataFrame([{
    "rd_spend": 125000.0,
    "admin_spend": np.nan,
    "marketing_spend": 260000.0
}])

pred_profit = production_pipeline.predict(raw_payload)[0]
print(f"Incoming Record: {raw_payload.to_dict(orient='records')[0]}")
print(f"Predicted Startup Profit: ${pred_profit:,.2f}")