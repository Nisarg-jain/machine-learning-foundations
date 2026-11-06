import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("=" * 65)
print("FEATURE ENGINEERING: OUTLIER DETECTION VIA Z-SCORE (3-SIGMA)")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC PLACEMENT DATASET
# -----------------------------------------------------------
np.random.seed(42)
n_students = 1000

# Feature 1: CGPA (Normal / Bell-shaped: Mean=6.96, Std=0.61)
cgpa = np.random.normal(loc=6.96, scale=0.61, size=n_students)

# Feature 2: Placement Exam Marks (Right-Skewed / Gamma-distributed)
exam_marks = np.random.gamma(shape=2.5, scale=12.0, size=n_students).clip(0, 100)

# Inject extreme synthetic anomalies into CGPA
outlier_indices = [45, 120, 310, 560, 890]
cgpa[outlier_indices[:3]] = [9.85, 9.92, 9.78]   # Extreme upper outliers
cgpa[outlier_indices[3:]] = [4.15, 3.90]         # Extreme lower outliers

# Target: Placed (1 = Hired, 0 = Not Hired)
logits = -5.5 + (0.85 * cgpa) + (0.015 * exam_marks)
probs = 1 / (1 + np.exp(-logits))
placed = np.random.binomial(n=1, p=probs, size=n_students)

df = pd.DataFrame({
    "cgpa": cgpa,
    "placement_exam_marks": exam_marks,
    "placed": placed
})

print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns.")

# -----------------------------------------------------------
# 2. AUDITING NORMALITY ASSUMPTION
# -----------------------------------------------------------
print("\n--- 2. Auditing Normality Assumption (Skewness) ---")
cgpa_skew = df["cgpa"].skew()
exam_skew = df["placement_exam_marks"].skew()

print(f"CGPA Skewness:                 {cgpa_skew:.3f} (Symmetric / Gaussian -> Valid for Z-Score)")
print(f"Placement Exam Marks Skewness: {exam_skew:.3f} (Significantly Skewed -> Invalid for Z-Score)")

# Leakage-Free Train-Test Split
X = df[["cgpa", "placement_exam_marks"]]
y = df["placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 3. Z-SCORE BOUNDARY AUDIT ON TRAINING PARTITION
# -----------------------------------------------------------
print("\n--- 3. Z-Score Boundary Audit (Strictly on X_train) ---")

mu_train = X_train["cgpa"].mean()
sigma_train = X_train["cgpa"].std()

upper_limit = mu_train + (3 * sigma_train)
lower_limit = mu_train - (3 * sigma_train)

print(f"Training Mean (μ):    {mu_train:.3f}")
print(f"Training Std (σ):     {sigma_train:.3f}")
print(f"Lower Bound (μ - 3σ): {lower_limit:.3f}")
print(f"Upper Bound (μ + 3σ): {upper_limit:.3f}")

# Identify outliers in training fold
train_outliers = X_train[(X_train["cgpa"] > upper_limit) | (X_train["cgpa"] < lower_limit)]
print(f"Detected Outliers in Training Fold: {len(train_outliers)} records ({len(train_outliers) / len(X_train):.2%})")

# -----------------------------------------------------------
# 4. PRODUCTION SCIKIT-LEARN TRANSFORMER: ZSCORECAPPER
# -----------------------------------------------------------
class ZScoreCapper(BaseEstimator, TransformerMixin):
    """
    Clamps values exceeding [μ - factor*σ, μ + factor*σ].
    Learns μ and σ exclusively during fit() to prevent data leakage.
    """
    def __init__(self, columns=None, factor=3.0):
        self.columns = columns
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        cols_to_fit = self.columns if self.columns is not None else X_df.columns
        
        self.bounds_ = {}
        for col in cols_to_fit:
            mean = X_df[col].mean()
            std = X_df[col].std()
            self.bounds_[col] = {
                "lower": mean - (self.factor * std),
                "upper": mean + (self.factor * std)
            }
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, bound in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower=bound["lower"], upper=bound["upper"])
        return X_df.values

# -----------------------------------------------------------
# 5. PIPELINE INTEGRATION & GENERALIZATION BENCHMARK
# -----------------------------------------------------------
print("\n--- 5. Benchmarking Model Generalization Performance ---")

# Pipeline A: Baseline without outlier capping
pipe_baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=42))
])
cv_baseline = cross_val_score(pipe_baseline, X_train, y_train, cv=5, scoring="accuracy").mean()
pipe_baseline.fit(X_train, y_train)
acc_baseline = pipe_baseline.score(X_test, y_test)

# Pipeline B: With ZScoreCapper inside the pipeline
pipe_capped = Pipeline([
    ("capper", ZScoreCapper(columns=["cgpa"], factor=3.0)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=42))
])
cv_capped = cross_val_score(pipe_capped, X_train, y_train, cv=5, scoring="accuracy").mean()
pipe_capped.fit(X_train, y_train)
acc_capped = pipe_capped.score(X_test, y_test)

print(f"Raw Baseline Pipeline:   5-Fold CV = {cv_baseline * 100:.2f}% | Holdout Test = {acc_baseline * 100:.2f}%")
print(f"Capped Z-Score Pipeline: 5-Fold CV = {cv_capped * 100:.2f}% | Holdout Test = {acc_capped * 100:.2f}%")

# -----------------------------------------------------------
# 6. VISUALIZATION DIAGNOSTICS
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: CGPA Bell Curve with 3-Sigma Boundaries
X_train["cgpa"].plot(kind="kde", ax=axes[0], color="navy", lw=2.5, label="CGPA Distribution")
axes[0].axvline(lower_limit, color="crimson", linestyle="--", lw=1.8, label=f"Lower (μ-3σ): {lower_limit:.2f}")
axes[0].axvline(upper_limit, color="crimson", linestyle="--", lw=1.8, label=f"Upper (μ+3σ): {upper_limit:.2f}")
axes[0].axvline(mu_train, color="gray", linestyle=":", lw=1.2, label=f"Mean (μ): {mu_train:.2f}")
axes[0].set_title("Gaussian CGPA with 3-Sigma Boundaries")
axes[0].set_xlabel("CGPA")
axes[0].legend()

# Plot 2: Before vs. After Capping Comparison
X_train_capped = pd.DataFrame(
    pipe_capped.named_steps["capper"].transform(X_train),
    columns=X_train.columns
)
X_train["cgpa"].plot(kind="kde", ax=axes[1], color="crimson", lw=2, linestyle="--", label="Before Capping (Tails Extended)")
X_train_capped["cgpa"].plot(kind="kde", ax=axes[1], color="forestgreen", lw=2, label="After Capping (Clamped to ±3σ)")
axes[1].set_title("Distribution Preservation Post-Capping")
axes[1].set_xlabel("CGPA")
axes[1].legend()

plt.tight_layout()
plt.savefig("visualizations/zscore_outlier_treatment.png")
plt.close()
print("\nSaved diagnostic figure -> 'visualizations/zscore_outlier_treatment.png'")

# -----------------------------------------------------------
# 7. INFERENCE TEST ON EXTREME UNPROCESSED INPUT
# -----------------------------------------------------------
print("\n--- 7. Inference on Extreme Unseen Record ---")

unseen_record = pd.DataFrame([{
    "cgpa": 11.5,                 # Invalid extreme outlier
    "placement_exam_marks": 78.0
}])

pred = pipe_capped.predict(unseen_record)[0]
prob = pipe_capped.predict_proba(unseen_record)[0][1]

status = "Placed" if pred == 1 else "Not Placed"
print(f"Incoming Record: {unseen_record.to_dict(orient='records')[0]}")
print(f"Model Prediction: {status} (Placement Probability: {prob:.2%})")