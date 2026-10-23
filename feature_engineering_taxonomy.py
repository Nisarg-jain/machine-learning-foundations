import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("=" * 65)
print("FEATURE ENGINEERING: THE 4-PILLAR WORKFLOW")
print("=" * 65)

# Load raw Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ===========================================================
# 1. PILLAR 1: FEATURE TRANSFORMATION
# ===========================================================
print("\n--- Pillar 1: Feature Transformation ---")

# A. Missing Value Imputation (Median for continuous, Mode for categorical)
median_age = df["Age"].median()
df["Age_imputed"] = df["Age"].fillna(median_age)
print(f"Age imputed via median ({median_age:.1f}). Missing remaining: {df['Age_imputed'].isnull().sum()}")

# B. Categorical Encoding (One-Hot Encoding for 'Sex')
encoder = OneHotEncoder(sparse_output=False, drop="first")
sex_encoded = encoder.fit_transform(df[["Sex"]])
df["Sex_encoded"] = sex_encoded
print(f"Encoded 'Sex' to binary numeric: {encoder.categories_[0]} -> [0, 1]")

# C. Feature Scaling (StandardScaler to mu=0, sigma=1)
scaler = StandardScaler()
df["Fare_scaled"] = scaler.fit_transform(df[["Fare"]])
print(f"Scaled 'Fare': Mean={df['Fare_scaled'].mean():.2f}, Std={df['Fare_scaled'].std():.2f}")

# ===========================================================
# 2. PILLAR 2: FEATURE CONSTRUCTION
# ===========================================================
print("\n--- Pillar 2: Feature Construction (Domain Driven) ---")

# Aggregate SibSp (siblings/spouse) and Parch (parents/children) into Family Size
df["family_size"] = df["SibSp"] + df["Parch"] + 1

# Segment into discrete behavioral travel groups
conditions = [
    (df["family_size"] == 1),
    (df["family_size"] >= 2) & (df["family_size"] <= 4),
    (df["family_size"] > 4)
]
choices = ["Alone", "Small_Family", "Large_Family"]
df["family_category"] = np.select(conditions, choices, default="Unknown")

print("Engineered 'family_size' and 'family_category' distributions:")
print(df[["family_size", "family_category"]].value_counts().head(5))

# ===========================================================
# 3. PILLAR 3: FEATURE SELECTION
# ===========================================================
print("\n--- Pillar 3: Feature Selection ---")

# Drop zero-signal identifiers and redundant raw columns
uninformative_cols = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch", "Age", "Fare", "Sex"]
df_selected = df.drop(columns=uninformative_cols)

print(f"Pruned {len(uninformative_cols)} raw/redundant columns.")
print(f"Remaining active feature set ({df_selected.shape[1]} cols): {list(df_selected.columns)}")

# ===========================================================
# 4. PILLAR 4: FEATURE EXTRACTION (PREVIEW)
# ===========================================================
print("\n--- Pillar 4: Feature Extraction Preview ---")
print("Dimensionality reduction (e.g., PCA) projects high-dimensional matrices")
print("into orthogonal principal components while retaining maximal variance.")

# Preview final transformed tabular format
print("\nFinal Processed Feature Sample:")
print(df_selected.head(3))