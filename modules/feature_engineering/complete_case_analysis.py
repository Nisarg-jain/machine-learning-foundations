import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 65)
print("FEATURE ENGINEERING: COMPLETE CASE ANALYSIS (CCA) AUDIT")
print("=" * 65)

# -----------------------------------------------------------
# 1. LOAD CANDIDATE RECRUITMENT DATASET
# -----------------------------------------------------------
# Features:
# - city_development_index: Continuous numerical (Low missingness ~2%)
# - training_hours: Continuous numerical (Low missingness ~4%)
# - education_level: Categorical (Low missingness ~2.5%)
# - company_size: Categorical (Heavy missingness ~35% -> Not CCA eligible)
# - target: Binary recruitment decision (0 = Not hired, 1 = Hired)

np.random.seed(42)
n_samples = 2000

city_dev = np.random.beta(a=5, b=2, size=n_samples)
train_hrs = np.random.exponential(scale=50, size=n_samples) + 5
edu_levels = np.random.choice(["Graduate", "Masters", "High School", "PhD"], size=n_samples, p=[0.70, 0.20, 0.08, 0.02])
comp_size = np.random.choice(["Small (<50)", "Medium (50-250)", "Large (250+)", np.nan], size=n_samples, p=[0.25, 0.25, 0.15, 0.35])
target = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])

# Inject missing values (MCAR pattern)
mask_city = np.random.rand(n_samples) < 0.02
mask_hrs = np.random.rand(n_samples) < 0.04
mask_edu = np.random.rand(n_samples) < 0.025

city_dev[mask_city] = np.nan
train_hrs[mask_hrs] = np.nan
edu_levels = np.array(edu_levels, dtype=object)
edu_levels[mask_edu] = np.nan

df = pd.DataFrame({
    "city_dev_index": city_dev,
    "training_hours": train_hrs,
    "education_level": edu_levels,
    "company_size": comp_size,
    "target": target
})

print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} features.")

# -----------------------------------------------------------
# 2. MISSINGNESS AUDIT & CCA ELIGIBILITY
# -----------------------------------------------------------
print("\n--- 2. Missingness Percentage Audit ---")
missing_summary = pd.DataFrame({
    "Null_Count": df.isnull().sum(),
    "Null_Percentage": (df.isnull().mean() * 100).round(2)
})
print(missing_summary)

# Identify features with 0% < missingness <= 5%
cca_candidates = missing_summary[
    (missing_summary["Null_Percentage"] > 0) & 
    (missing_summary["Null_Percentage"] <= 5.0)
].index.tolist()

print(f"\nFeatures eligible for Complete Case Analysis (< 5% nulls):")
print(f"  {cca_candidates}")

# -----------------------------------------------------------
# 3. APPLYING LISTWISE DELETION (CCA)
# -----------------------------------------------------------
# Drop rows only where the CCA candidate columns contain NaN
df_cca = df.dropna(subset=cca_candidates).copy()

data_retained_pct = (len(df_cca) / len(df)) * 100
print(f"\nRow count before CCA: {len(df)}")
print(f"Row count after CCA:  {len(df_cca)}")
print(f"Total dataset retained: {data_retained_pct:.2f}% (Safe threshold: >= 90%)")

# -----------------------------------------------------------
# 4. STATISTICAL VALIDATION: NUMERICAL DISTRIBUTIONS
# -----------------------------------------------------------
print("\n--- 4. Numerical Distribution Parity Check ---")

num_cols = ["city_dev_index", "training_hours"]
for col in num_cols:
    orig_mean, cca_mean = df[col].mean(), df_cca[col].mean()
    orig_std, cca_std = df[col].std(), df_cca[col].std()
    print(f"Column '{col}':")
    print(f"  Mean -> Original: {orig_mean:.4f} | CCA: {cca_mean:.4f} (Shift: {abs(orig_mean - cca_mean):.4f})")
    print(f"  Std  -> Original: {orig_std:.4f} | CCA: {cca_std:.4f} (Shift: {abs(orig_std - cca_std):.4f})")

# Visual KDE Verification
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, col in enumerate(num_cols):
    df[col].plot(kind="kde", ax=axes[idx], color="crimson", lw=2, label="Original (Pre-CCA)")
    df_cca[col].plot(kind="kde", ax=axes[idx], color="forestgreen", lw=2, linestyle="--", label="Complete Cases (Post-CCA)")
    axes[idx].set_title(f"KDE Density Check: {col}")
    axes[idx].set_xlabel(col)
    axes[idx].legend()

plt.tight_layout()
plt.savefig("visualizations/cca_distribution_audit.png")
plt.close()
print("Saved distribution verification plot -> 'visualizations/cca_distribution_audit.png'")

# -----------------------------------------------------------
# 5. STATISTICAL VALIDATION: CATEGORICAL PROPORTIONS
# -----------------------------------------------------------
print("\n--- 5. Categorical Proportion Stability Check ---")

cat_col = "education_level"
ratio_orig = df[cat_col].value_counts(normalize=True)
ratio_cca = df_cca[cat_col].value_counts(normalize=True)

cat_comparison = pd.DataFrame({
    "Original_Ratio": ratio_orig,
    "Post_CCA_Ratio": ratio_cca
})
cat_comparison["Absolute_Diff"] = (cat_comparison["Original_Ratio"] - cat_comparison["Post_CCA_Ratio"]).abs()
print(cat_comparison.round(4))