import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("=" * 65)
print("FEATURE ENGINEERING: RANDOM SAMPLE IMPUTATION & MISSING INDICATOR")
print("=" * 65)

# -----------------------------------------------------------
# 1. LOAD BENCHMARK DATASET (TITANIC REPLICA)
# -----------------------------------------------------------
np.random.seed(42)
n_samples = 1000

# Continuous features
age = np.random.normal(loc=30, scale=12, size=n_samples).clip(1, 75)
fare = np.random.exponential(scale=35, size=n_samples) + 7.5

# Inject ~20% missingness into Age
mask_age = np.random.rand(n_samples) < 0.20
age_missing = age.copy()
age_missing[mask_age] = np.nan

# Target: Survival probability (older passengers had different survival odds)
logits = -1.2 + (0.025 * age) + (0.015 * fare) + (1.5 * mask_age.astype(int))
probs = 1 / (1 + np.exp(-logits))
survived = np.random.binomial(1, probs)

df = pd.DataFrame({
    "age": age_missing,
    "fare": fare,
    "survived": survived
})

print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
print(f"Missing Age Count: {df['age'].isna().sum()} ({df['age'].isna().mean():.1%})")

# Leakage-free train-test split
X = df[["age", "fare"]]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 2. RANDOM SAMPLE IMPUTATION MECHANICS
# -----------------------------------------------------------
print("\n--- 2. Performing Random Sample Imputation ---")

# Extract observed pool strictly from training data
observed_age_pool = X_train["age"].dropna().values

# Impute Training Set
X_train_rsi = X_train.copy()
missing_train_count = X_train_rsi["age"].isna().sum()

# Draw random values with a fixed seed for reproducibility
rng = np.random.RandomState(42)
random_replacements_train = rng.choice(observed_age_pool, size=missing_train_count, replace=True)
X_train_rsi.loc[X_train_rsi["age"].isna(), "age"] = random_replacements_train

# Compare with standard Mean Imputation
age_mean = X_train["age"].mean()
X_train_mean = X_train.copy()
X_train_mean["age"] = X_train_mean["age"].fillna(age_mean)

# Audit Variance
orig_var = X_train["age"].dropna().var()
mean_var = X_train_mean["age"].var()
rsi_var = X_train_rsi["age"].var()

print(f"Variance -> Original Observed Age: {orig_var:.2f}")
print(f"Variance -> Mean Imputation:       {mean_var:.2f} (Collapses by ~20%!)")
print(f"Variance -> Random Sample:         {rsi_var:.2f} (Nearly identical to original)")

# -----------------------------------------------------------
# 3. VISUALIZING DISTRIBUTION INTEGRITY
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))

X_train["age"].dropna().plot(kind="kde", ax=ax, color="black", lw=2.5, label="Original (Observed)")
X_train_mean["age"].plot(kind="kde", ax=ax, color="crimson", lw=2, linestyle="--", label="Mean Imputed (Spike)")
X_train_rsi["age"].plot(kind="kde", ax=ax, color="forestgreen", lw=2, linestyle=":", label="Random Sample (Preserved)")

ax.set_title("Distribution Preservation: Mean vs. Random Sample Imputation")
ax.set_xlabel("Age")
ax.legend()
plt.tight_layout()
plt.savefig("visualizations/random_sample_vs_mean.png")
plt.close()
print("Saved comparison curve -> 'visualizations/random_sample_vs_mean.png'")

# -----------------------------------------------------------
# 4. THE MISSING INDICATOR & PIPELINE ORCHESTRATION
# -----------------------------------------------------------
print("\n--- 4. Benchmarking Model Accuracy With & Without Missing Indicator ---")

# Model 1: Baseline SimpleImputer (Mean, No Indicator)
pipe_no_indicator = Pipeline([
    ("imputer", SimpleImputer(strategy="mean", add_indicator=False)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=42))
])
pipe_no_indicator.fit(X_train, y_train)
acc_no_ind = pipe_no_indicator.score(X_test, y_test)

# Model 2: SimpleImputer (Mean WITH add_indicator=True)
pipe_with_indicator = Pipeline([
    ("imputer", SimpleImputer(strategy="mean", add_indicator=True)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=42))
])
pipe_with_indicator.fit(X_train, y_train)
acc_ind = pipe_with_indicator.score(X_test, y_test)

print(f"Model Accuracy WITHOUT Missing Indicator: {acc_no_ind * 100:.2f}%")
print(f"Model Accuracy WITH Missing Indicator:    {acc_ind * 100:.2f}%")
print(f"Lift from capturing the missingness signal: +{(acc_ind - acc_no_ind) * 100:.2f}%")

# Inspect the features created by the pipeline
imputer_step = pipe_with_indicator.named_steps["imputer"]
print(f"\nMissing indicator features detected by Scikit-Learn: Column Index {imputer_step.indicator_.features_}")

# -----------------------------------------------------------
# 5. INFERENCE ON RAW USER RECORD
# -----------------------------------------------------------
print("\n--- 5. Direct Inference on Unprocessed User Record ---")

incoming_payload = pd.DataFrame([{
    "age": np.nan,    # Missing age will automatically be imputed AND flagged
    "fare": 85.0
}])

pred = pipe_with_indicator.predict(incoming_payload)[0]
prob = pipe_with_indicator.predict_proba(incoming_payload)[0][1]

status = "Survived" if pred == 1 else "Deceased"
print(f"Incoming Record: {incoming_payload.to_dict(orient='records')[0]}")
print(f"Prediction: {status} (Survival Probability: {prob:.2%})")