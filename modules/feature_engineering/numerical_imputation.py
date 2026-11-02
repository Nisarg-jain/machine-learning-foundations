import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

print("=" * 65)
print("FEATURE ENGINEERING: UNIVARIATE NUMERICAL IMPUTATION")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC BENCHMARK DATASET
# -----------------------------------------------------------
np.random.seed(42)
n_samples = 1000

# Base continuous features
age = np.random.normal(loc=30, scale=12, size=n_samples).clip(1, 80)
fare = np.random.exponential(scale=35, size=n_samples) + 5
family_size = np.random.poisson(lam=1.2, size=n_samples)

# Target: binary survival indicator
logits = -1.2 + (0.03 * age) + (0.015 * fare) - (0.4 * family_size)
prob = 1 / (1 + np.exp(-logits))
survived = np.random.binomial(n=1, p=prob, size=n_samples)

df = pd.DataFrame({
    "age": age,
    "fare": fare,
    "family_size": family_size,
    "survived": survived
})

# Inject missing values:
# - age: 20% missingness (Heavy missingness -> prone to severe distortion)
# - fare: 5% missingness (Light missingness -> minimal distortion)
mask_age = np.random.rand(n_samples) < 0.20
mask_fare = np.random.rand(n_samples) < 0.05

df_missing = df.copy()
df_missing.loc[mask_age, "age"] = np.nan
df_missing.loc[mask_fare, "fare"] = np.nan

print(f"Data Loaded: {df_missing.shape[0]} rows, {df_missing.shape[1]} columns.")
print(f"Missing Values -> age: {df_missing['age'].isna().sum()} ({df_missing['age'].isna().mean():.1%}), "
      f"fare: {df_missing['fare'].isna().sum()} ({df_missing['fare'].isna().mean():.1%})")

# -----------------------------------------------------------
# 2. TRAIN-TEST SPLIT (LEAKAGE PREVENTION)
# -----------------------------------------------------------
X = df_missing.drop(columns=["survived"])
y = df_missing["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 3. STATISTICAL IMPUTATION AUDIT (AGE COLUMN)
# -----------------------------------------------------------
print("\n--- 3. Statistical Distortion Audit (Age Column) ---")

train_age_orig = X_train["age"].dropna()
mean_val = X_train["age"].mean()
median_val = X_train["age"].median()

# End-of-distribution values
std_val = X_train["age"].std()
eod_val = mean_val + 3 * std_val  # μ + 3σ for Gaussian

# Imputed series
age_mean_imp = X_train["age"].fillna(mean_val)
age_median_imp = X_train["age"].fillna(median_val)
age_eod_imp = X_train["age"].fillna(eod_val)
age_arbitrary_imp = X_train["age"].fillna(-1)

# Audit A: Variance Shrinkage
print(f"Variance -> Original: {train_age_orig.var():.2f}")
print(f"Variance -> Mean Imp: {age_mean_imp.var():.2f} (Notice variance shrinkage)")
print(f"Variance -> Median Imp: {age_median_imp.var():.2f}")
print(f"Variance -> EOD Imp:    {age_eod_imp.var():.2f} (Variance expands due to tail constant)")

# Audit B: Correlation with other features
corr_orig = train_age_orig.corr(X_train.loc[train_age_orig.index, "family_size"])
corr_mean = age_mean_imp.corr(X_train["family_size"])
corr_median = age_median_imp.corr(X_train["family_size"])

print(f"\nCorrelation with 'family_size':")
print(f"  Original:   {corr_orig:.4f}")
print(f"  Mean Imp:   {corr_mean:.4f} (Diluted towards 0)")
print(f"  Median Imp: {corr_median:.4f}")

# Audit C: Tukey IQR Outlier Boundaries
def compute_iqr_limits(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

lower_orig, upper_orig = compute_iqr_limits(train_age_orig)
lower_med, upper_med = compute_iqr_limits(age_median_imp)

print(f"\nTukey Outlier Thresholds for Age:")
print(f"  Original Data:      [{lower_orig:.2f}, {upper_orig:.2f}]")
print(f"  Post-Median Impute: [{lower_med:.2f}, {upper_med:.2f}] (Boundaries contracted)")

# -----------------------------------------------------------
# 4. DIAGNOSTIC PLOTS (KDE & BOXPLOTS)
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: KDE Density Overlays
train_age_orig.plot(kind="kde", ax=axes[0], color="black", lw=2.5, label="Original (Pre-Impute)")
age_mean_imp.plot(kind="kde", ax=axes[0], color="red", lw=2, linestyle="--", label="Mean Imputed (Spike)")
age_median_imp.plot(kind="kde", ax=axes[0], color="green", lw=2, linestyle=":", label="Median Imputed")
axes[0].set_title("Age: PDF Distortion & Central Spike")
axes[0].set_xlabel("Age")
axes[0].legend()

# Plot 2: Boxplot Comparison
box_df = pd.DataFrame({
    "Original": train_age_orig,
    "Median_Imp": age_median_imp
})
box_df.plot(kind="box", ax=axes[1], patch_artist=True)
axes[1].set_title("Age: IQR Contraction & False Outliers")
axes[1].set_ylabel("Age")

plt.tight_layout()
plt.savefig("visualizations/numerical_imputation_diagnostics.png")
plt.close()
print("\nSaved diagnostics figure -> 'visualizations/numerical_imputation_diagnostics.png'")

# -----------------------------------------------------------
# 5. SCIKIT-LEARN PRODUCTION PIPELINE INTEGRATION
# -----------------------------------------------------------
print("\n--- 5. Orchestrating with Scikit-Learn Pipeline ---")

# Define strategy per column:
# - age is approximately Gaussian -> strategy='mean' (or 'median' if skewed)
# - fare is heavily skewed -> strategy='median'
preprocessor = ColumnTransformer(
    transformers=[
        ("age_imputer", SimpleImputer(strategy="mean"), ["age"]),
        ("fare_imputer", SimpleImputer(strategy="median"), ["fare"])
    ],
    remainder="passthrough"  # family_size passes through untouched
)

production_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42))
])

# Fit pipeline strictly on raw training DataFrame (learns mean of age and median of fare)
production_pipeline.fit(X_train, y_train)

# Evaluate on raw test set containing nulls
test_score = production_pipeline.score(X_test, y_test)
print(f"Pipeline fitted successfully.")
print(f"Test Set Accuracy: {test_score * 100:.2f}%")

# Inspect learned statistics
fitted_age_mean = production_pipeline.named_steps["preprocessor"].named_transformers_["age_imputer"].statistics_[0]
fitted_fare_median = production_pipeline.named_steps["preprocessor"].named_transformers_["fare_imputer"].statistics_[0]

print(f"\nLearned Training Statistics:")
print(f"  Age Mean:     {fitted_age_mean:.2f}")
print(f"  Fare Median:  {fitted_fare_median:.2f}")

# -----------------------------------------------------------
# 6. INFERENCE TEST ON RAW PRODUCTION PAYLOAD
# -----------------------------------------------------------
print("\n--- 6. Direct Inference on Unprocessed User Record ---")

raw_payload = pd.DataFrame([{
    "age": np.nan,        # Missing age handled via training mean (fitted_age_mean)
    "fare": np.nan,       # Missing fare handled via training median (fitted_fare_median)
    "family_size": 2
}])

prediction = production_pipeline.predict(raw_payload)[0]
prob = production_pipeline.predict_proba(raw_payload)[0][1]

status = "Survived" if prediction == 1 else "Deceased"
print(f"Incoming Record: {raw_payload.to_dict(orient='records')[0]}")
print(f"Pipeline Prediction: {status} (Survival Probability: {prob:.2%})")