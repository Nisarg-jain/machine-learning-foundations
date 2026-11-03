import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("=" * 65)
print("FEATURE ENGINEERING: ADVANCED UNIVARIATE IMPUTATION")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC BENCHMARK DATASET (TITANIC REPLICA)
# -----------------------------------------------------------
np.random.seed(42)
n_samples = 1200

# Continuous features
age = np.random.normal(loc=30, scale=13, size=n_samples).clip(1, 80)
fare = np.random.exponential(scale=32, size=n_samples) + 7.5

# Categorical features
embarked = np.random.choice(["S", "C", "Q"], size=n_samples, p=[0.70, 0.20, 0.10]).astype(object)
pclass = np.random.choice([1, 2, 3], size=n_samples, p=[0.25, 0.25, 0.50])

# Inject missing values:
# Age: 20% MNAR (older people slightly less likely to report)
mask_age = (np.random.rand(n_samples) + (age / 120.0)) > 0.90
age_missing = age.copy()
age_missing[mask_age] = np.nan

# Embarked: 5% MCAR missingness
mask_emb = np.random.rand(n_samples) < 0.05
embarked_missing = embarked.copy()
embarked_missing[mask_emb] = np.nan

# Binary Target: Survived (heavily conditioned on missing age signal + pclass)
logits = -1.5 + (0.02 * age) - (0.8 * (pclass - 1)) + (0.015 * fare) + (1.2 * mask_age.astype(int))
probs = 1 / (1 + np.exp(-logits))
survived = np.random.binomial(n=1, p=probs, size=n_samples)

df = pd.DataFrame({
    "age": age_missing,
    "fare": fare,
    "pclass": pclass,
    "embarked": embarked_missing,
    "survived": survived
})

print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} features.")
print(f"Missing Values -> Age: {df['age'].isna().sum()} ({df['age'].isna().mean():.1%}), "
      f"Embarked: {df['embarked'].isna().sum()} ({df['embarked'].isna().mean():.1%})")

# Leakage-Free Train-Test Split
X = df.drop(columns=["survived"])
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 2. CUSTOM SCIKIT-LEARN RANDOM SAMPLE IMPUTER
# -----------------------------------------------------------
class RandomSampleImputer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compliant transformer that replaces missing values
    by drawing randomly from observed training samples.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.train_samples_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.train_samples_ = {}
        for col in X_df.columns:
            observed = X_df[col].dropna().values
            self.train_samples_[col] = observed
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        rng = np.random.RandomState(self.random_state)
        
        for col in X_df.columns:
            null_mask = X_df[col].isna()
            null_count = null_mask.sum()
            if null_count > 0:
                observed = self.train_samples_[col]
                # Draw random replacements strictly from learned training pool
                imputed_values = rng.choice(observed, size=null_count, replace=True)
                X_df.loc[null_mask, col] = imputed_values
        return X_df.values

# -----------------------------------------------------------
# 3. STATISTICAL COMPARISON: MEDIAN VS. RANDOM SAMPLE
# -----------------------------------------------------------
print("\n--- 3. Distribution Audit: Median vs. Random Sample ---")

orig_age_train = X_train["age"].dropna()

# Median Imputation
median_val = X_train["age"].median()
age_median_imp = X_train["age"].fillna(median_val)

# Random Sample Imputation
rsi = RandomSampleImputer(random_state=42)
age_rsi_imp = pd.Series(rsi.fit_transform(X_train[["age"]])[:, 0], index=X_train.index)

print(f"Variance -> Original Observed Age: {orig_age_train.var():.2f}")
print(f"Variance -> Post-Median Impute:    {age_median_imp.var():.2f} (Variance deflated)")
print(f"Variance -> Post-Random Sample:    {age_rsi_imp.var():.2f} (Variance fully preserved!)")

# Plot KDEs
fig, ax = plt.subplots(figsize=(9, 5))
orig_age_train.plot(kind="kde", ax=ax, color="black", lw=2.5, label="Original Observed")
age_median_imp.plot(kind="kde", ax=ax, color="red", lw=2, linestyle="--", label="Median Imputed (Spike)")
age_rsi_imp.plot(kind="kde", ax=ax, color="green", lw=2, linestyle="-.", label="Random Sample (Preserved)")
ax.set_title("Age Distribution: Median vs. Random Sample Imputation")
ax.set_xlabel("Age")
ax.legend()
plt.tight_layout()
plt.savefig("visualizations/random_sample_vs_median.png")
plt.close()
print("Saved comparison plot -> 'visualizations/random_sample_vs_median.png'")

# -----------------------------------------------------------
# 4. MEASURING THE LIFT OF MISSING INDICATORS
# -----------------------------------------------------------
print("\n--- 4. Empirical Evaluation: The Impact of Missing Indicators ---")

# Model A: Baseline SimpleImputer (No indicator)
pipe_baseline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(random_state=42))
])
pipe_baseline.fit(X_train[["age", "fare"]], y_train)
acc_baseline = accuracy_score(y_test, pipe_baseline.predict(X_test[["age", "fare"]]))

# Model B: SimpleImputer WITH add_indicator=True
pipe_indicator = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(random_state=42))
])
pipe_indicator.fit(X_train[["age", "fare"]], y_train)
acc_indicator = accuracy_score(y_test, pipe_indicator.predict(X_test[["age", "fare"]]))

print(f"Accuracy WITHOUT Missing Indicator: {acc_baseline * 100:.2f}%")
print(f"Accuracy WITH Missing Indicator:    {acc_indicator * 100:.2f}% (Measurable lift from MNAR signal)")

# -----------------------------------------------------------
# 5. AUTOMATED PIPELINE SEARCH WITH GRIDSEARCHCV
# -----------------------------------------------------------
print("\n--- 5. Automated Hyperparameter Tuning of Imputation Strategies ---")

numeric_features = ["age", "fare"]
categorical_features = ["embarked", "pclass"]

# Construct Modular ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler())
        ]), numeric_features),
        
        ("cat", Pipeline([
            ("imputer", SimpleImputer()),
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=500, random_state=42))
])

# Define the parameter search space across imputation strategies
param_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "preprocessor__num__imputer__add_indicator": [True, False],
    "preprocessor__cat__imputer__strategy": ["most_frequent", "constant"],
    "preprocessor__cat__imputer__fill_value": ["Missing"],
    "classifier__C": [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Optimal Configuration Discovered:")
for param, val in grid_search.best_params_.items():
    print(f"  {param}: {val}")

print(f"Best 5-Fold Cross-Validation Accuracy: {grid_search.best_score_ * 100:.2f}%")

final_test_acc = grid_search.score(X_test, y_test)
print(f"Holdout Test Generalization Accuracy: {final_test_acc * 100:.2f}%")