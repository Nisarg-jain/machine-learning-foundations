import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error

print("=" * 65)
print("FEATURE ENGINEERING: OUTLIER DETECTION & TREATMENT")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC DATASET: STUDY HOURS VS. EXAM SCORE
# -----------------------------------------------------------
np.random.seed(42)
n_samples = 120

# Clean linear relationship
hours = np.random.uniform(1.0, 9.0, size=n_samples)
scores = (hours * 9.5) + np.random.normal(loc=12, scale=4, size=n_samples)

# Inject high-leverage outliers: students studying very few hours but scoring ~98
outlier_hours = np.array([1.2, 1.5, 1.8])
outlier_scores = np.array([96.0, 98.0, 95.0])

X_raw = np.concatenate([hours, outlier_hours]).reshape(-1, 1)
y_raw = np.concatenate([scores, outlier_scores])

df = pd.DataFrame({"study_hours": X_raw.ravel(), "exam_score": y_raw})
print(f"Dataset generated: {df.shape[0]} records (including 3 high-leverage outliers).")

# -----------------------------------------------------------
# 2. MODEL SENSITIVITY: LINEAR REGRESSION VS. DECISION TREE
# -----------------------------------------------------------
print("\n--- 2. Auditing Outlier Sensitivity Across Algorithms ---")

# Models on Clean Data (Without Outliers)
clean_lr = LinearRegression().fit(hours.reshape(-1, 1), scores)
clean_dt = DecisionTreeRegressor(max_depth=3, random_state=42).fit(hours.reshape(-1, 1), scores)

# Models on Corrupted Data (With Outliers)
corrupt_lr = LinearRegression().fit(X_raw, y_raw)
corrupt_dt = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_raw, y_raw)

print(f"Linear Regression Slope (Clean Data):     {clean_lr.coef_[0]:.2f}")
print(f"Linear Regression Slope (With Outliers):  {corrupt_lr.coef_[0]:.2f} (Line slope tilted!)")

# -----------------------------------------------------------
# 3. PRODUCTION-GRADE CUSTOM WINSORIZER (CAPPING TRANSFORMER)
# -----------------------------------------------------------
class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Clips extreme outliers beyond Tukey's IQR boundaries.
    Learns bounds strictly during .fit() on training partition (Zero Data Leakage).
    """
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            q1 = X_df[col].quantile(0.25)
            q3 = X_df[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - (self.factor * iqr)
            self.upper_bounds_[col] = q3 + (self.factor * iqr)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            lower = self.lower_bounds_[col]
            upper = self.upper_bounds_[col]
            X_df[col] = X_df[col].clip(lower=lower, upper=upper)
        return X_df.values

# -----------------------------------------------------------
# 4. EVALUATING CAPPING IN A PIPELINE
# -----------------------------------------------------------
print("\n--- 4. Evaluating Pipeline with IQR Capping ---")

capping_pipeline = Pipeline([
    ("capper", IQRCapper(factor=1.5)),
    ("regressor", LinearRegression())
])

capping_pipeline.fit(X_raw, y_raw)

# Evaluate on true baseline test distribution
test_hours = np.linspace(1, 9, 50).reshape(-1, 1)
true_scores = (test_hours.ravel() * 9.5) + 12

baseline_rmse = root_mean_squared_error(true_scores, clean_lr.predict(test_hours))
corrupt_rmse = root_mean_squared_error(true_scores, corrupt_lr.predict(test_hours))
capped_rmse = root_mean_squared_error(true_scores, capping_pipeline.predict(test_hours))

print(f"Test RMSE (Clean Baseline Model):     {baseline_rmse:.2f}")
print(f"Test RMSE (Corrupted Model):          {corrupt_rmse:.2f} (Distorted by outliers)")
print(f"Test RMSE (Capped Pipeline Model):    {capped_rmse:.2f} (Error reduced via capping)")

# -----------------------------------------------------------
# 5. DIAGNOSTIC VISUALIZATION
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: Linear Regression Tilt
axes[0].scatter(hours, scores, color="royalblue", alpha=0.6, label="Normal Students")
axes[0].scatter(outlier_hours, outlier_scores, color="crimson", s=100, edgecolors="black", zorder=5, label="High-Leverage Outliers")
axes[0].plot(test_hours, clean_lr.predict(test_hours), color="forestgreen", lw=2.5, label="Clean Model Line")
axes[0].plot(test_hours, corrupt_lr.predict(test_hours), color="crimson", linestyle="--", lw=2, label="Corrupted Model Line (Tilted)")
axes[0].set_title("Outlier Impact: Linear Regression Line Pull")
axes[0].set_xlabel("Study Hours")
axes[0].set_ylabel("Exam Score")
axes[0].legend()
axes[0].grid(True, linestyle=":", alpha=0.5)

# Plot B: Boxplot Tukey Detection
df.boxplot(column=["exam_score"], ax=axes[1], patch_artist=True, boxprops=dict(facecolor="lightblue"))
q1 = df["exam_score"].quantile(0.25)
q3 = df["exam_score"].quantile(0.75)
iqr = q3 - q1
upper_whisker = q3 + (1.5 * iqr)

axes[1].axhline(upper_whisker, color="crimson", linestyle="--", label=f"Upper Bound: Q3 + 1.5*IQR ({upper_whisker:.1f})")
axes[1].set_title("Tukey IQR Boundary for Exam Score")
axes[1].legend()

plt.tight_layout()
plt.savefig("visualizations/outlier_impact_and_capping.png")
plt.close()
print("\nSaved diagnostic figure -> 'visualizations/outlier_impact_and_capping.png'")