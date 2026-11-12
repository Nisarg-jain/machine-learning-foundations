import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

print("=" * 65)
print("FEATURE ENGINEERING: OUTLIER DETECTION & TREATMENT (IQR METHOD)")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC PLACEMENT DATASET (MIXED DISTRIBUTIONS)
# -----------------------------------------------------------
np.random.seed(42)
n_students = 1000

# Feature 1: CGPA (Symmetric / Gaussian -> Skewness near 0)
cgpa = np.random.normal(loc=6.96, scale=0.61, size=n_students)
cgpa[[45, 120, 310]] = [9.85, 9.92, 9.78]
cgpa[[560, 890]] = [4.15, 3.90]

# Feature 2: Placement Exam Marks (Right-Skewed -> Skewness > 0.8)
# Replicating Q1 ≈ 17, Median ≈ 28, Q3 ≈ 44, Max = 100
exam_marks = np.random.gamma(shape=2.1, scale=15.0, size=n_students).clip(0, 100)

# Target: Placed (1 = Hired, 0 = Not Hired)
logits = -5.5 + (0.85 * cgpa) + (0.018 * exam_marks)
probs = 1 / (1 + np.exp(-logits))
placed = np.random.binomial(n=1, p=probs, size=n_students)

df = pd.DataFrame({
    "cgpa": cgpa,
    "placement_exam_marks": exam_marks,
    "placed": placed
})

print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns.")
print(f"CGPA Skewness:                 {df['cgpa'].skew():.3f} (Gaussian -> Z-Score Candidate)")
print(f"Placement Exam Marks Skewness: {df['placement_exam_marks'].skew():.3f} (Right-Skewed -> IQR Candidate)")

# Leakage-Free Train-Test Split
X = df[["cgpa", "placement_exam_marks"]]
y = df["placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 2. IQR QUARTILE & FENCE AUDIT (STRICTLY ON X_TRAIN)
# -----------------------------------------------------------
print("\n--- 2. Tukey IQR Proximity Audit (X_train) ---")

q1 = X_train["placement_exam_marks"].quantile(0.25)
median = X_train["placement_exam_marks"].quantile(0.50)
q3 = X_train["placement_exam_marks"].quantile(0.75)
iqr = q3 - q1

upper_fence = q3 + (1.5 * iqr)
lower_fence = q1 - (1.5 * iqr)

print(f"25th Percentile (Q1):    {q1:.2f}")
print(f"50th Percentile (Med):   {median:.2f}")
print(f"75th Percentile (Q3):    {q3:.2f}")
print(f"Interquartile Range:     {iqr:.2f}")
print(f"Lower Fence (Q1-1.5IQR): {lower_fence:.2f} (Below min 0 -> no lower outliers)")
print(f"Upper Fence (Q3+1.5IQR): {upper_fence:.2f}")

outlier_mask = (X_train["placement_exam_marks"] > upper_fence) | (X_train["placement_exam_marks"] < lower_fence)
print(f"Outliers Detected in X_train: {outlier_mask.sum()} records ({outlier_mask.mean():.2%})")

# -----------------------------------------------------------
# 3. VISUALIZING TRIMMING VS. CAPPING DIAGNOSTICS
# -----------------------------------------------------------
os.makedirs("visualizations", exist_ok=True)

# Strategy A: Trimming (Row Removal)
X_train_trimmed = X_train[~outlier_mask].copy()

# Strategy B: Capping / Winsorization (Clamping to Fences)
X_train_capped = X_train.copy()
X_train_capped["placement_exam_marks"] = X_train_capped["placement_exam_marks"].clip(
    lower=lower_fence, upper=upper_fence
)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Row 1: Trimming Comparison
X_train["placement_exam_marks"].plot(kind="kde", ax=axes[0, 0], color="crimson", lw=2, label="Original")
X_train_trimmed["placement_exam_marks"].plot(kind="kde", ax=axes[0, 0], color="navy", lw=2, linestyle="--", label="Trimmed")
axes[0, 0].axvline(upper_fence, color="black", linestyle=":", label=f"Upper Fence ({upper_fence:.1f})")
axes[0, 0].set_title("Trimming: KDE Distribution Comparison")
axes[0, 0].legend()

axes[0, 1].boxplot(
    [X_train["placement_exam_marks"], X_train_trimmed["placement_exam_marks"]],
    tick_labels=["Original", "Post-Trimming"],
    patch_artist=True,
    boxprops=dict(facecolor="lightsteelblue")
)
axes[0, 1].set_title("Trimming: Boxplot Comparison (Note Recalculated Whiskers)")

# Row 2: Capping Comparison
X_train["placement_exam_marks"].plot(kind="kde", ax=axes[1, 0], color="crimson", lw=2, label="Original")
X_train_capped["placement_exam_marks"].plot(kind="kde", ax=axes[1, 0], color="forestgreen", lw=2, linestyle="--", label="Capped (Winsorized)")
axes[1, 0].axvline(upper_fence, color="black", linestyle=":", label=f"Upper Fence ({upper_fence:.1f})")
axes[1, 0].set_title("Capping: KDE Comparison (Note Tail Bump at Cap)")
axes[1, 0].legend()

axes[1, 1].boxplot(
    [X_train["placement_exam_marks"], X_train_capped["placement_exam_marks"]],
    tick_labels=["Original", "Post-Capping"],
    patch_artist=True,
    boxprops=dict(facecolor="lightgreen")
)
axes[1, 1].set_title("Capping: Boxplot Comparison (All Outliers Clamped)")

plt.tight_layout()
plt.savefig("visualizations/iqr_outlier_treatment.png")
plt.close()
print("\nSaved diagnostic plots -> 'visualizations/iqr_outlier_treatment.png'")

# -----------------------------------------------------------
# 4. PRODUCTION SCIKIT-LEARN TRANSFORMERS (IQR + Z-SCORE)
# -----------------------------------------------------------
class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Caps outliers using Tukey's IQR Proximity Rule [Q1 - factor*IQR, Q3 + factor*IQR].
    Learns quartile fences strictly during .fit() to prevent data leakage.
    """
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.bounds_ = {}
        for col in X_df.columns:
            q1_val = X_df[col].quantile(0.25)
            q3_val = X_df[col].quantile(0.75)
            iqr_val = q3_val - q1_val
            self.bounds_[col] = {
                "lower": q1_val - (self.factor * iqr_val),
                "upper": q3_val + (self.factor * iqr_val)
            }
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, bound in self.bounds_.items():
            X_df[col] = X_df[col].clip(lower=bound["lower"], upper=bound["upper"])
        return X_df.values


class ZScoreCapper(BaseEstimator, TransformerMixin):
    """
    Caps outliers using [μ - factor*σ, μ + factor*σ] for Gaussian features.
    """
    def __init__(self, factor=3.0):
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.bounds_ = {}
        for col in X_df.columns:
            mean_val = X_df[col].mean()
            std_val = X_df[col].std()
            self.bounds_[col] = {
                "lower": mean_val - (self.factor * std_val),
                "upper": mean_val + (self.factor * std_val)
            }
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, bound in self.bounds_.items():
            X_df[col] = X_df[col].clip(lower=bound["lower"], upper=bound["upper"])
        return X_df.values

# -----------------------------------------------------------
# 5. HYBRID DISTRIBUTION-AWARE COLUMNTRANSFORMER PIPELINE
# -----------------------------------------------------------
print("\n--- 5. Building Distribution-Aware Preprocessing Pipeline ---")

# Route Gaussian columns to ZScoreCapper and Skewed columns to IQRCapper
outlier_preprocessor = ColumnTransformer(
    transformers=[
        ("gaussian_capper", ZScoreCapper(factor=3.0), ["cgpa"]),
        ("skewed_capper", IQRCapper(factor=1.5), ["placement_exam_marks"])
    ]
)

full_pipeline = Pipeline([
    ("outlier_treatment", outlier_preprocessor),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=42))
])

cv_acc = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()
full_pipeline.fit(X_train, y_train)
test_acc = full_pipeline.score(X_test, y_test)

print(f"Hybrid Pipeline 5-Fold CV Accuracy: {cv_acc * 100:.2f}%")
print(f"Hybrid Pipeline Holdout Accuracy:   {test_acc * 100:.2f}%")

# -----------------------------------------------------------
# 6. INFERENCE ON EXTREME UNSEEN PAYLOAD
# -----------------------------------------------------------
print("\n--- 6. Direct Inference on Unprocessed Extreme Record ---")

unseen_student = pd.DataFrame([{
    "cgpa": 10.5,                     # Exceeds Z-score upper bound -> clamped
    "placement_exam_marks": 99.0      # Exceeds IQR upper fence -> clamped
}])

pred = full_pipeline.predict(unseen_student)[0]
prob = full_pipeline.predict_proba(unseen_student)[0][1]

status = "Placed" if pred == 1 else "Not Placed"
print(f"Incoming Raw Record: {unseen_student.to_dict(orient='records')[0]}")
print(f"Pipeline Prediction: {status} (Placement Probability: {prob:.2%})")