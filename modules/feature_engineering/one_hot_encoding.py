import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

print("=" * 65)
print("FEATURE ENGINEERING: ONE-HOT ENCODING & DUMMY VARIABLE TRAP")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC AUTOMOTIVE SALES DATASET
# -----------------------------------------------------------
# Features:
# - brand: High-cardinality nominal category
# - km_driven: Continuous numerical
# - fuel: Low-cardinality nominal category (Diesel, Petrol, CNG, LPG)
# - owner: Low-cardinality nominal category (First, Second, Third)
# - selling_price: Continuous numerical target

np.random.seed(42)
n_samples = 500

brands_pool = ["Maruti", "Hyundai", "Mahindra", "Tata", "Honda", "Toyota", "Ford", "BMW", "Audi", "Volvo", "Skoda"]
# Skewed brand distribution (Maruti & Hyundai dominate; Volvo & Audi are rare)
brand_probs = [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.03, 0.015, 0.01, 0.01, 0.005]

df = pd.DataFrame({
    "brand": np.random.choice(brands_pool, size=n_samples, p=brand_probs),
    "km_driven": np.random.randint(10000, 150000, size=n_samples),
    "fuel": np.random.choice(["Petrol", "Diesel", "CNG", "LPG"], size=n_samples, p=[0.48, 0.46, 0.04, 0.02]),
    "owner": np.random.choice(["First Owner", "Second Owner", "Third Owner"], size=n_samples, p=[0.65, 0.25, 0.10]),
    "selling_price": np.random.randint(200000, 1200000, size=n_samples)
})

print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head(4))

# -----------------------------------------------------------
# 2. DEMONSTRATING THE DUMMY VARIABLE TRAP
# -----------------------------------------------------------
print("\n--- 2. Dummy Variable Trap & Multicollinearity ---")

# Naive encoding using all K columns
naive_dummies = pd.get_dummies(df["fuel"], prefix="fuel", dtype=int)
print("Naive K-Columns Dummy Encoding (First 3 rows):")
print(naive_dummies.head(3))
print("Sum across all columns for any observation:")
print(naive_dummies.head(3).sum(axis=1).values)
print("-> Perfect Multicollinearity: Any column = 1 - sum(other columns).")

# Solution: Drop the first column to retain K - 1 linearly independent columns
k_minus_1_dummies = pd.get_dummies(df["fuel"], prefix="fuel", drop_first=True, dtype=int)
print("\nRemedied K-1 Columns (drop_first=True):")
print(k_minus_1_dummies.head(3))

# -----------------------------------------------------------
# 3. SCIKIT-LEARN OneHotEncoder IN PRODUCTION PIPELINES
# -----------------------------------------------------------
print("\n--- 3. Scikit-Learn OneHotEncoder Workflow ---")

X = df[["fuel", "owner", "km_driven"]]
y = df["selling_price"]

# Leakage-Free Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Initialize encoder:
# - drop='first' avoids the dummy variable trap for linear estimators
# - sparse_output=False returns dense numpy arrays directly
ohe = OneHotEncoder(drop="first", sparse_output=False, dtype=np.int32)

# Fit strictly on categorical columns of the training set
categorical_cols = ["fuel", "owner"]
ohe.fit(X_train[categorical_cols])

X_train_encoded = ohe.transform(X_train[categorical_cols])
X_test_encoded = ohe.transform(X_test[categorical_cols])

# Fetch automatically synthesized feature names
feature_names = ohe.get_feature_names_out(categorical_cols)
X_train_encoded_df = pd.DataFrame(
    X_train_encoded, columns=feature_names, index=X_train.index
)

print(f"Original categorical columns: {categorical_cols}")
print(f"Generated K-1 indicator columns ({len(feature_names)}): {list(feature_names)}")
print("\nSample encoded training rows:")
print(X_train_encoded_df.head(3))

# -----------------------------------------------------------
# 4. HANDLING UNSEEN TEST CATEGORIES (handle_unknown='ignore')
# -----------------------------------------------------------
print("\n--- 4. Unseen Inference Categories Handling ---")

# In production inference, new categories may appear (e.g., Electric vehicle)
# When drop=None, handle_unknown='ignore' outputs all zeros for novel categories
robust_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
robust_ohe.fit(X_train[["fuel"]])

novel_sample = pd.DataFrame({"fuel": ["Electric"]})
encoded_novel = robust_ohe.transform(novel_sample)
print(f"Inference input with unknown category 'Electric':")
print(f"Encoded representation: {encoded_novel[0]} (All zeros, pipeline survives without crashing)")

# -----------------------------------------------------------
# 5. HIGH-CARDINALITY HANDLING (TOP-N FREQUENCY THRESHOLDING)
# -----------------------------------------------------------
print("\n--- 5. High-Cardinality Pruning: Top-N Thresholding ---")

brand_counts = df["brand"].value_counts()
print("Raw Category Counts for 'brand':")
print(brand_counts)

# Define frequency threshold: retain brands with >= 30 records, group the rest as 'Other'
threshold = 30
frequent_brands = brand_counts[brand_counts >= threshold].index
print(f"\nBrands meeting threshold (>={threshold} records): {list(frequent_brands)}")

df_pruned = df.copy()
df_pruned["brand_binned"] = df_pruned["brand"].apply(
    lambda x: x if x in frequent_brands else "Other"
)

print("\nCardinality Reduction:")
print(f"  Original distinct brands: {df['brand'].nunique()}")
print(f"  Pruned distinct brands:   {df_pruned['brand_binned'].nunique()}")
print("New category distribution:")
print(df_pruned["brand_binned"].value_counts())