import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

print("=" * 65)
print("FEATURE ENGINEERING: SCIKIT-LEARN COLUMN TRANSFORMER")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC CLINICAL INTAKE DATASET
# -----------------------------------------------------------
# Columns:
# - age: Continuous numeric (needs no processing / passthrough)
# - gender: Nominal categorical (Male, Female)
# - fever: Continuous numeric with missing NaN values (needs Imputation)
# - cough: Ordinal categorical (Mild < Strong) (needs Ordinal Encoding)
# - city: Nominal categorical (Kolkata, Bangalore, Delhi, Mumbai) (needs One-Hot)
# - has_covid: Classification target label

np.random.seed(42)
n_patients = 100

age = np.random.randint(18, 75, size=n_patients)
gender = np.random.choice(["Male", "Female"], size=n_patients)
fever = np.random.choice([98.0, 99.2, 100.5, 101.4, 102.8, 103.5, np.nan], size=n_patients, p=[0.2, 0.2, 0.2, 0.15, 0.1, 0.05, 0.1])
cough = np.random.choice(["Mild", "Strong"], size=n_patients, p=[0.6, 0.4])
city = np.random.choice(["Kolkata", "Bangalore", "Delhi", "Mumbai"], size=n_patients)
has_covid = np.random.choice(["No", "Yes"], size=n_patients, p=[0.65, 0.35])

df = pd.DataFrame({
    "age": age,
    "gender": gender,
    "fever": fever,
    "cough": cough,
    "city": city,
    "has_covid": has_covid
})

print(f"Dataset generated: {df.shape[0]} patients, {df.shape[1]} columns.")
print(f"Missing fever entries: {df['fever'].isnull().sum()} nulls.")
print("\nRaw Sample:")
print(df.head(4))

# -----------------------------------------------------------
# 2. TRAIN-TEST SPLIT (LEAKAGE PREVENTION)
# -----------------------------------------------------------
X = df.drop(columns=["has_covid"])
y = df["has_covid"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

# ===========================================================
# 3. THE MANUAL METHOD (SLICING & CONCATENATION)
# ===========================================================
print("\n--- 3. Manual Slicing & Stacking (Verbose Baseline) ---")

# Step A: Impute Fever
imputer = SimpleImputer(strategy="mean")
train_fever_imp = imputer.fit_transform(X_train[["fever"]])

# Step B: Ordinal Encode Cough (Mild: 0, Strong: 1)
ord_enc = OrdinalEncoder(categories=[["Mild", "Strong"]])
train_cough_enc = ord_enc.fit_transform(X_train[["cough"]])

# Step C: One-Hot Encode Gender & City (drop='first' to prevent dummy trap)
ohe_enc = OneHotEncoder(drop="first", sparse_output=False)
train_ohe_enc = ohe_enc.fit_transform(X_train[["gender", "city"]])

# Step D: Extract untouched Age
train_age = X_train[["age"]].values

# Step E: Horizontal Concatenation
X_train_manual = np.concatenate(
    [train_fever_imp, train_cough_enc, train_ohe_enc, train_age], axis=1
)
print(f"Manually constructed array shape: {X_train_manual.shape}")

# ===========================================================
# 4. THE PRODUCTION METHOD (ColumnTransformer)
# ===========================================================
print("\n--- 4. Clean Orchestration via ColumnTransformer ---")

# Define the parallel multi-transformer pipeline
transformer = ColumnTransformer(
    transformers=[
        # Tuple 1: Impute missing fever values with mean
        ("fever_imputer", SimpleImputer(strategy="mean"), ["fever"]),
        # Tuple 2: Ordinal encode cough with explicit hierarchy
        ("cough_encoder", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["cough"]),
        # Tuple 3: One-hot encode nominal features with dummy trap mitigation
        ("nominal_encoder", OneHotEncoder(drop="first", sparse_output=False), ["gender", "city"])
    ],
    # Preserve unmentioned columns (age) without dropping them
    remainder="passthrough"
)

# Fit on training data and transform in a single call
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

print(f"ColumnTransformer X_train shape: {X_train_transformed.shape}")
print(f"ColumnTransformer X_test shape:  {X_test_transformed.shape}")

# -----------------------------------------------------------
# 5. VERIFICATION: NUMERICAL PARITY CHECK
# -----------------------------------------------------------
print("\n--- 5. Numerical Parity Audit ---")
is_identical = np.allclose(X_train_manual, X_train_transformed)
print(f"Do Manual Slicing and ColumnTransformer outputs match exactly? -> {is_identical}")

# Display synthesized feature names
feature_names = transformer.get_feature_names_out()
print("\nSynthesized Column Order in Output Matrix:")
for idx, name in enumerate(feature_names):
    print(f"  Col {idx}: {name}")

transformed_sample_df = pd.DataFrame(
    X_train_transformed[:3], columns=feature_names
)
print("\nFirst 3 Transformed Rows:")
print(transformed_sample_df)