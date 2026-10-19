import pandas as pd
import numpy as np

print("=" * 65)
print("AUTOMATED DATA AUDIT & UNDERSTANDING PROFILER")
print("=" * 65)

# 1. Generate a mock messy dataset for auditing
np.random.seed(42)
n_rows = 120

data = {
    "student_id": np.arange(1001, 1001 + n_rows),
    "cgpa": np.random.normal(7.2, 0.9, n_rows).round(2),
    "iq": np.random.randint(80, 135, n_rows),
    "internships": np.random.choice([0, 1, 2, 3], size=n_rows, p=[0.4, 0.35, 0.2, 0.05]),
    "branch": np.random.choice(["CSE", "IT", "ECE", "MECH"], size=n_rows),
    "placement": np.random.choice([0, 1], size=n_rows, p=[0.35, 0.65])
}

df_audit = pd.DataFrame(data)

# Inject dirty elements: missing values, outliers, duplicate rows
df_audit.loc[5:10, "cgpa"] = np.nan
df_audit.loc[20, "iq"] = 280  # Obvious outlier
df_audit = pd.concat([df_audit, df_audit.iloc[0:4]], ignore_index=True)  # Injected duplicates

# 2. Step-by-Step Profiling Audit
print(f"1. Dataset Shape: {df_audit.shape[0]} rows, {df_audit.shape[1]} columns")

print("\n2. Duplicate Check:")
duplicates_count = df_audit.duplicated().sum()
print(f"   Found {duplicates_count} duplicate rows.")

print("\n3. Missing Value Analysis:")
missing_series = df_audit.isnull().sum()
missing_pct = (df_audit.isnull().sum() / len(df_audit)) * 100
missing_report = pd.DataFrame({"Missing Count": missing_series, "Percentage (%)": missing_pct.round(2)})
print(missing_report[missing_report["Missing Count"] > 0])

print("\n4. Five-Number Summary & Distribution Metrics:")
print(df_audit.describe().round(2))

print("\n5. Categorical Feature Cardinality:")
for col in df_audit.select_dtypes(include=["object"]).columns:
    print(f"   Column '{col}': {df_audit[col].nunique()} unique categories -> {df_audit[col].unique()}")

print("\n6. Target Correlation Ranking:")
numeric_corr = df_audit.corr(numeric_only=True)
print(numeric_corr["placement"].sort_values(ascending=False).round(3))