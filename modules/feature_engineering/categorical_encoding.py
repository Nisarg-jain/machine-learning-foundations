import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

print("=" * 65)
print("FEATURE ENGINEERING: ORDINAL & LABEL ENCODING PIPELINE")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC CUSTOMER FEEDBACK DATASET
# -----------------------------------------------------------
# Features:
# - age: Continuous numeric
# - gender: Nominal categorical (Male, Female)
# - review: Ordinal categorical (Poor, Average, Good)
# - education: Ordinal categorical (School, UG, PG)
# - purchased: Target label (No, Yes)

data = {
    "age": [28, 34, 45, 23, 52, 38, 41, 29, 31, 55],
    "gender": ["Female", "Male", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
    "review": ["Good", "Poor", "Average", "Good", "Poor", "Good", "Average", "Poor", "Good", "Average"],
    "education": ["UG", "School", "PG", "School", "UG", "PG", "School", "UG", "PG", "School"],
    "purchased": ["No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "Yes"]
}

df = pd.DataFrame(data)
print("\nRaw Dataset Sample:")
print(df.to_string(index=False))

# -----------------------------------------------------------
# 2. FEATURE ISOLATION & LEAKAGE-FREE TRAIN/TEST SPLIT
# -----------------------------------------------------------
# Extract Ordinal input features for X
X = df[["review", "education"]]
# Extract 1D target vector for y
y = df["purchased"]

# Split prior to applying any transformation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTraining Set Instances: {X_train.shape[0]} rows")
print(f"Testing Set Instances:  {X_test.shape[0]} rows")

# -----------------------------------------------------------
# 3. ORDINAL ENCODING (FOR 2D INPUT MATRIX X)
# -----------------------------------------------------------
print("\n--- Applying OrdinalEncoder to Feature Matrix X ---")

# Define the explicit domain hierarchy for each ordinal feature:
# Feature 0 (review): Poor (0) < Average (1) < Good (2)
# Feature 1 (education): School (0) < UG (1) < PG (2)
review_order = ["Poor", "Average", "Good"]
education_order = ["School", "UG", "PG"]

ordinal_encoder = OrdinalEncoder(categories=[review_order, education_order])

# Learn categories strictly on training partition
ordinal_encoder.fit(X_train)

# Transform both train and test partitions
X_train_encoded = ordinal_encoder.transform(X_train)
X_test_encoded = ordinal_encoder.transform(X_test)

# Convert to formatted DataFrames for clear inspection
X_train_encoded_df = pd.DataFrame(
    X_train_encoded, columns=X_train.columns, index=X_train.index
)
X_test_encoded_df = pd.DataFrame(
    X_test_encoded, columns=X_test.columns, index=X_test.index
)

print("Learned Ordinal Hierarchies:")
for idx, cats in enumerate(ordinal_encoder.categories_):
    print(f"  Column '{X.columns[idx]}': {list(cats)} -> {list(range(len(cats)))}")

print("\nTransformed X_train (Ordinal Encoded):")
print(pd.concat([X_train, X_train_encoded_df.add_suffix("_encoded")], axis=1))

# -----------------------------------------------------------
# 4. LABEL ENCODING (STRICTLY FOR 1D TARGET VECTOR y)
# -----------------------------------------------------------
print("\n--- Applying LabelEncoder to Target Vector y ---")

label_encoder = LabelEncoder()

# Fit strictly on target training partition
label_encoder.fit(y_train)

# Transform train and test ground truths
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print(f"Learned Target Classes: {list(label_encoder.classes_)} -> [0, 1]")

y_train_comparison = pd.DataFrame({
    "Original Target (y_train)": y_train,
    "Encoded Label (y_train_enc)": y_train_encoded
})
print("\nTransformed y_train:")
print(y_train_comparison.to_string(index=False))

# -----------------------------------------------------------
# 5. INVERSE TRANSFORM VALIDATION (INTEGRITY CHECK)
# -----------------------------------------------------------
print("\n--- Verification: Inverse Transformations ---")
reconstructed_X = ordinal_encoder.inverse_transform(X_test_encoded)
reconstructed_y = label_encoder.inverse_transform(y_test_encoded)

print("Reconstructed X_test matches original:", np.array_equal(reconstructed_X, X_test.values))
print("Reconstructed y_test matches original:", np.array_equal(reconstructed_y, y_test.values))