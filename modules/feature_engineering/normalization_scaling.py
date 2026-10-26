import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    StandardScaler
)

print("=" * 65)
print("FEATURE SCALING: NORMALIZATION TECHNIQUES COMPARISON")
print("=" * 65)

# Synthetic wine dataset features: Alcohol (11-15%) & Malic Acid (1-6 g/L)
np.random.seed(42)
n_samples = 300

alcohol = np.random.uniform(11.0, 14.8, size=n_samples)
malic_acid = np.random.exponential(scale=1.5, size=n_samples) + 0.8

# Inject 3 extreme sensor anomalies / outliers
alcohol[0] = 25.0
malic_acid[0] = 18.0
malic_acid[1] = 16.5

df = pd.DataFrame({
    "Alcohol": alcohol,
    "MalicAcid": malic_acid
})

print(f"Dataset generated: {df.shape[0]} rows with injected outliers.")
print(df.head(4).round(2))

# -----------------------------------------------------------
# 1. TRAIN / TEST SPLIT (PREVENT DATA LEAKAGE)
# -----------------------------------------------------------
X_train, X_test = train_test_split(df, test_size=0.25, random_state=42)

# -----------------------------------------------------------
# 2. FIT SCALERS ON TRAIN ONLY
# -----------------------------------------------------------
minmax = MinMaxScaler()
maxabs = MaxAbsScaler()
robust = RobustScaler()
standard = StandardScaler()

X_train_minmax = minmax.fit_transform(X_train)
X_train_maxabs = maxabs.fit_transform(X_train)
X_train_robust = robust.fit_transform(X_train)
X_train_standard = standard.fit_transform(X_train)

# -----------------------------------------------------------
# 3. STATISTICAL COMPARISON TABLE
# -----------------------------------------------------------
def extract_stats(scaled_array, name):
    return {
        "Scaler": name,
        "Min (Alcohol)": round(scaled_array[:, 0].min(), 3),
        "Max (Alcohol)": round(scaled_array[:, 0].max(), 3),
        "Median (Alcohol)": round(np.median(scaled_array[:, 0]), 3),
        "Min (MalicAcid)": round(scaled_array[:, 1].min(), 3),
        "Max (MalicAcid)": round(scaled_array[:, 1].max(), 3),
        "Median (MalicAcid)": round(np.median(scaled_array[:, 1]), 3)
    }

comparison_records = [
    extract_stats(X_train.values, "Raw Unscaled"),
    extract_stats(X_train_minmax, "MinMaxScaler [0, 1]"),
    extract_stats(X_train_maxabs, "MaxAbsScaler [-1, 1]"),
    extract_stats(X_train_standard, "StandardScaler (Z-Score)"),
    extract_stats(X_train_robust, "RobustScaler (IQR Based)")
]

summary_df = pd.DataFrame(comparison_records)
print("\n--- 3. COMPARATIVE SCALING METRICS ---")
print(summary_df.to_string(index=False))

# -----------------------------------------------------------
# 4. ZERO-PRESERVATION AUDIT (SPARSE DATA TEST)
# -----------------------------------------------------------
print("\n--- 4. ZERO PRESERVATION AUDIT (SPARSE SIMULATION) ---")
sparse_sample = np.array([
    [0.0, 12.0],
    [0.0, 0.0],
    [5.0, 0.0]
])

sparse_minmax = minmax.fit_transform(sparse_sample)
sparse_maxabs = maxabs.fit_transform(sparse_sample)
sparse_standard = standard.fit_transform(sparse_sample)

print("Original Sparse Sample (Note structural zeros):")
print(sparse_sample)
print("\nMaxAbsScaler (Zeros preserved strictly):")
print(sparse_maxabs)
print("\nStandardScaler (Zeros destroyed by mean-shift):")
print(sparse_standard.round(2))

# -----------------------------------------------------------
# 5. OUTLIER SENSITIVITY TEST
# -----------------------------------------------------------
print("\n--- 5. OUTLIER SENSITIVITY DEMONSTRATION ---")
outlier_point = pd.DataFrame({"Alcohol": [25.0], "MalicAcid": [18.0]})

print("Outlier Point Scaled Output:")
print(f"  MinMaxScaler:   {minmax.transform(outlier_point)[0].round(3)} (Pushes core data into tiny sub-range)")
print(f"  RobustScaler:   {robust.transform(outlier_point)[0].round(3)} (Retains realistic scale for core data)")