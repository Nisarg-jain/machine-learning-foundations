import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

print("=" * 65)
print("FEATURE TRANSFORMATION: FUNCTIONTRANSFORMER & NORMALITY")
print("=" * 65)

# -----------------------------------------------------------
# 1. LOAD TITANIC BENCHMARK DATASET
# -----------------------------------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Select features: Age (mostly normal), Fare (heavy right-skew), Survived (target)
features = ["Age", "Fare"]
X = df[features].copy()
y = df["Survived"].copy()

# Leakage-Free Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Impute median age to ensure a complete numerical matrix
imputer = SimpleImputer(strategy="median")
X_train["Age"] = imputer.fit_transform(X_train[["Age"]])
X_test["Age"] = imputer.transform(X_test[["Age"]])

print(f"Training Instances: {X_train.shape[0]} rows")
print("\n--- Initial Distribution Metrics ---")
print(f"Age Skewness:  {X_train['Age'].skew():.4f} (Close to 0 -> approximately symmetric)")
print(f"Fare Skewness: {X_train['Fare'].skew():.4f} (Heavy positive skew -> right tail)")

# -----------------------------------------------------------
# 2. GENERATE DIAGNOSTIC Q-Q PLOTS
# -----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Age KDE & Q-Q
axes[0, 0].set_title("Age: Kernel Density Distribution")
X_train["Age"].plot(kind="kde", ax=axes[0, 0], color="navy", lw=2)
axes[0, 0].set_xlabel("Age")

stats.probplot(X_train["Age"], dist="norm", plot=axes[0, 1])
axes[0, 1].set_title("Age: Normal Q-Q Plot (Empirical vs. Theoretical)")

# Fare KDE & Q-Q (Raw)
axes[1, 0].set_title("Fare: Kernel Density Distribution (Raw)")
X_train["Fare"].plot(kind="kde", ax=axes[1, 0], color="crimson", lw=2)
axes[1, 0].set_xlabel("Fare ($)")

stats.probplot(X_train["Fare"], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Fare: Normal Q-Q Plot (Raw Right Skew)")

plt.tight_layout()
plt.savefig("visualizations/distribution_normality_audit.png")
plt.close()
print("Saved diagnostic plot -> 'visualizations/distribution_normality_audit.png'")

# -----------------------------------------------------------
# 3. EMPIRICAL BENCHMARK: RAW VS. LOG-TRANSFORMED
# -----------------------------------------------------------
print("\n--- 3. Performance Benchmark (Raw vs. Log-Transformed) ---")

# Baseline: Evaluate models on raw distributions
raw_lr_cv = cross_val_score(LogisticRegression(random_state=42), X_train, y_train, cv=10, scoring="accuracy").mean()
raw_dt_cv = cross_val_score(DecisionTreeClassifier(random_state=42), X_train, y_train, cv=10, scoring="accuracy").mean()

print(f"Baseline (Raw Features):")
print(f"  Logistic Regression (10-Fold CV Mean): {raw_lr_cv * 100:.2f}%")
print(f"  Decision Tree       (10-Fold CV Mean): {raw_dt_cv * 100:.2f}%")

# Construct Log-Transformation Pipeline:
# np.log1p(x) = ln(1 + x) ensures fares of 0.0 don't cause -inf errors
log_transformer = FunctionTransformer(func=np.log1p)

# Apply log transform strictly to 'Fare', passing 'Age' through untouched
fare_log_ct = ColumnTransformer(
    transformers=[
        ("fare_log", log_transformer, ["Fare"])
    ],
    remainder="passthrough"
)

X_train_log = fare_log_ct.fit_transform(X_train)

log_lr_cv = cross_val_score(LogisticRegression(random_state=42), X_train_log, y_train, cv=10, scoring="accuracy").mean()
log_dt_cv = cross_val_score(DecisionTreeClassifier(random_state=42), X_train_log, y_train, cv=10, scoring="accuracy").mean()

print(f"\nPost Log-Transformation (np.log1p applied to Fare):")
print(f"  Logistic Regression (10-Fold CV Mean): {log_lr_cv * 100:.2f}% (Notice the measurable gain)")
print(f"  Decision Tree       (10-Fold CV Mean): {log_dt_cv * 100:.2f}% (Invariant to monotonic transform)")

# -----------------------------------------------------------
# 4. COMPARING MATHEMATICAL MAPPINGS ON SKEWED DATA
# -----------------------------------------------------------
print("\n--- 4. Comparing Transformation Mappings on Fare ---")

fare_values = X_train["Fare"].values

# Transforms
fare_log = np.log1p(fare_values)
fare_sqrt = np.sqrt(np.maximum(0, fare_values))
fare_reciprocal = 1.0 / (fare_values + 0.1)

print(f"Raw Fare Skewness:        {stats.skew(fare_values):.4f}")
print(f"Log1p Fare Skewness:      {stats.skew(fare_log):.4f} (Near 0 -> successfully normalized)")
print(f"Square Root Fare Skewness:{stats.skew(fare_sqrt):.4f}")
print(f"Reciprocal Fare Skewness: {stats.skew(fare_reciprocal):.4f}")

# Generate Post-Transformation Q-Q Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

stats.probplot(fare_log, dist="norm", plot=axes[0])
axes[0].set_title(f"Log1p Transform (Skew: {stats.skew(fare_log):.2f})")

stats.probplot(fare_sqrt, dist="norm", plot=axes[1])
axes[1].set_title(f"Square Root Transform (Skew: {stats.skew(fare_sqrt):.2f})")

stats.probplot(fare_reciprocal, dist="norm", plot=axes[2])
axes[2].set_title(f"Reciprocal Transform (Skew: {stats.skew(fare_reciprocal):.2f})")

plt.tight_layout()
plt.savefig("visualizations/fare_transforms_qq_comparison.png")
plt.close()
print("Saved comparison plot -> 'visualizations/fare_transforms_qq_comparison.png'")