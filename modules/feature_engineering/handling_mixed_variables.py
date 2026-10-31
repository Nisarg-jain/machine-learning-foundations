import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

print("=" * 65)
print("FEATURE ENGINEERING: HANDLING MIXED VARIABLES")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC MIXED-TYPE BENCHMARK DATASET
# -----------------------------------------------------------
# Simulating real-world customer intake with two mixed columns:
# - cabin_code: Composite strings ('C85', 'B22', 'E49', etc.)
# - dependents: Multi-type records (numbers '1', '2', mixed with 'Alone', 'Unknown')
# - monthly_charges: Clean continuous feature
# - churn: Binary classification target

np.random.seed(42)
n_records = 300

cabins = np.random.choice(["C85", "C123", "B22", "B58", "D12", "E49", "F33", np.nan], size=n_records, p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.15])
dependents_raw = np.random.choice(["1", "2", "3", "4", "Alone", "Unknown"], size=n_records, p=[0.35, 0.25, 0.10, 0.05, 0.20, 0.05])
charges = np.random.normal(loc=70, scale=25, size=n_records).clip(20, 150)
churn = np.random.choice([0, 1], size=n_records, p=[0.7, 0.3])

df = pd.DataFrame({
    "cabin_code": cabins,
    "dependents": dependents_raw,
    "monthly_charges": charges,
    "churn": churn
})

print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns.")
print("\nRaw Mixed Dataset Sample:")
print(df.head(6))

# -----------------------------------------------------------
# 2. PATTERN 1 MECHANICS: COMPOSITE TOKEN DECOMPOSITION
# -----------------------------------------------------------
print("\n--- Pattern 1: Decomposing Composite Tokens ('cabin_code') ---")

def parse_composite_tokens(df_subset):
    """
    Splits composite alphanumeric strings into:
    1. Deck class letter (Categorical)
    2. Room coordinate number (Numerical)
    """
    series = df_subset["cabin_code"].astype(str)
    
    # Extract leading alphabetic characters (e.g., 'C' from 'C85')
    deck = series.str.extract(r"([A-Za-z]+)", expand=False).fillna("Missing")
    
    # Extract numerical digits (e.g., '85' from 'C85')
    room_num = pd.to_numeric(series.str.extract(r"(\d+)", expand=False), errors="coerce")
    
    return pd.DataFrame({
        "cabin_deck": deck,
        "cabin_num": room_num
    }, index=df_subset.index)

parsed_cabin_sample = parse_composite_tokens(df.head(6))
print(pd.concat([df[["cabin_code"]].head(6), parsed_cabin_sample], axis=1))

# -----------------------------------------------------------
# 3. PATTERN 2 MECHANICS: MULTI-TYPE COERCION & SPLIT
# -----------------------------------------------------------
print("\n--- Pattern 2: Slicing Multi-Type Records ('dependents') ---")

def parse_multitype_column(df_subset):
    """
    Decouples a column containing both integer digits and categorical text.
    - Numerical: Numbers preserved, text coerced to NaN -> imputed.
    - Categorical: Text retained as specific category flag.
    """
    series = df_subset["dependents"]
    
    # Coerce text strings to NaN; valid numbers become float
    numeric_comp = pd.to_numeric(series, errors="coerce")
    
    # Identify non-numeric strings to capture categorical semantics
    categorical_comp = series.where(numeric_comp.isna(), other="Numeric_Entry")
    
    return pd.DataFrame({
        "dependents_num": numeric_comp,
        "dependents_flag": categorical_comp
    }, index=df_subset.index)

parsed_dependents_sample = parse_multitype_column(df.head(6))
print(pd.concat([df[["dependents"]].head(6), parsed_dependents_sample], axis=1))

# -----------------------------------------------------------
# 4. PRODUCTION-GRADE PIPELINE INTEGRATION
# -----------------------------------------------------------
print("\n--- 4. Building End-to-End Scikit-Learn Preprocessing Pipeline ---")

X = df.drop(columns=["churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def engineer_mixed_features(input_df):
    """
    Vectorized transformer function executing all mixed-variable extractions.
    Returns a unified clean DataFrame ready for standard scaling and encoding.
    """
    df_out = input_df.copy()
    
    # Decompose cabin_code
    cabin_series = df_out["cabin_code"].astype(str)
    df_out["cabin_deck"] = cabin_series.str.extract(r"([A-Za-z]+)", expand=False).fillna("Missing")
    df_out["cabin_num"] = pd.to_numeric(cabin_series.str.extract(r"(\d+)", expand=False), errors="coerce")
    
    # Decompose dependents
    dep_series = df_out["dependents"]
    df_out["dependents_num"] = pd.to_numeric(dep_series, errors="coerce")
    df_out["dependents_flag"] = dep_series.where(df_out["dependents_num"].isna(), other="Numeric")
    
    # Drop raw unparsed mixed columns
    df_out.drop(columns=["cabin_code", "dependents"], inplace=True)
    return df_out

# Wrap data wrangling step into a stateless Scikit-Learn FunctionTransformer
feature_extractor = FunctionTransformer(engineer_mixed_features)

# Define column categories for downstream transformations
clean_numeric_cols = ["monthly_charges", "cabin_num", "dependents_num"]
clean_categorical_cols = ["cabin_deck", "dependents_flag"]

# Downstream standard preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), clean_numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ]), clean_categorical_cols)
    ]
)

# Full End-to-End Pipeline
full_pipeline = Pipeline([
    ("mixed_feature_extraction", feature_extractor),
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train pipeline directly on raw mixed DataFrame
full_pipeline.fit(X_train, y_train)
test_accuracy = full_pipeline.score(X_test, y_test)

print(f"Pipeline successfully fitted on raw mixed inputs.")
print(f"Holdout Generalization Accuracy: {test_accuracy * 100:.2f}%")

# -----------------------------------------------------------
# 5. INFERENCE AUDIT ON RAW UNSTRUCTURED ROW
# -----------------------------------------------------------
print("\n--- 5. Direct Inference on Unprocessed User Request ---")

unseen_record = pd.DataFrame([{
    "cabin_code": "B77",
    "dependents": "Alone",
    "monthly_charges": 89.50
}])

pred = full_pipeline.predict(unseen_record)[0]
prob = full_pipeline.predict_proba(unseen_record)[0][1]

status = "Churn Risk" if pred == 1 else "Retained"
print(f"Incoming Raw Input: {unseen_record.to_dict(orient='records')[0]}")
print(f"Pipeline Output: {status} (Predicted Churn Probability: {prob:.2%})")