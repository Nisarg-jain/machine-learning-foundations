import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

print("=" * 65)
print("FEATURE ENGINEERING: CATEGORICAL MISSING DATA IMPUTATION")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC REAL ESTATE BENCHMARK DATASET
# -----------------------------------------------------------
np.random.seed(42)
n_samples = 1200

# garage_qual: Dominant 'Typical' class, low MCAR missingness (~4%)
garage_choices = ["Ex", "Gd", "TA", "Fa"]
garage_probs = [0.05, 0.15, 0.75, 0.05]
garage_qual = np.random.choice(garage_choices, size=n_samples, p=garage_probs).astype(object)
mask_garage = np.random.rand(n_samples) < 0.04
garage_qual[mask_garage] = np.nan

# fireplace_qual: High structural missingness (~48% of homes have NO fireplace)
fireplace_choices = ["Ex", "Gd", "TA", "Fa", "Po"]
fireplace_probs = [0.10, 0.40, 0.35, 0.10, 0.05]
fireplace_qual = np.random.choice(fireplace_choices, size=n_samples, p=fireplace_probs).astype(object)
mask_fireplace = np.random.rand(n_samples) < 0.48
fireplace_qual[mask_fireplace] = np.nan

# Continuous living area and target sale_price
gr_liv_area = np.random.normal(loc=1500, scale=400, size=n_samples).clip(600, 3500)
has_fireplace_bonus = (~pd.isna(fireplace_qual)).astype(int) * 35000
sale_price = (gr_liv_area * 110) + has_fireplace_bonus + np.random.normal(0, 15000, size=n_samples)

df = pd.DataFrame({
    "gr_liv_area": gr_liv_area,
    "garage_qual": garage_qual,
    "fireplace_qual": fireplace_qual,
    "sale_price": sale_price
})

print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns.")
print("\nMissingness Summary:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nProportion of nulls:")
print((df.isnull().mean() * 100).round(2).to_dict())

# -----------------------------------------------------------
# 2. TRAIN-TEST SPLIT (LEAKAGE PREVENTION)
# -----------------------------------------------------------
X = df.drop(columns=["sale_price"])
y = df["sale_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 3. STATISTICAL DISTORTION AUDIT: MODE VS. MISSING
# -----------------------------------------------------------
print("\n--- 3. Evaluating Mode Imputation Impact on High Missingness ---")

# Inspect raw training distribution of fireplace_qual
raw_counts = X_train["fireplace_qual"].value_counts(dropna=False, normalize=True)
mode_fireplace = X_train["fireplace_qual"].mode()[0]

# Simulate Mode Imputation on Fireplace Quality
fireplace_mode_imputed = X_train["fireplace_qual"].fillna(mode_fireplace)
mode_counts = fireplace_mode_imputed.value_counts(normalize=True)

# Simulate Explicit Missing Category Imputation
fireplace_cat_imputed = X_train["fireplace_qual"].fillna("Missing")
cat_counts = fireplace_cat_imputed.value_counts(normalize=True)

comparison_df = pd.DataFrame({
    "Original_Valid_Ratio": X_train["fireplace_qual"].value_counts(normalize=True),
    "Post_Mode_Impute": mode_counts,
    "Post_Missing_Category": cat_counts
}).fillna(0)

print(comparison_df.round(3))
print(f"\nMode category '{mode_fireplace}' artificially surged from {raw_counts.get(mode_fireplace, 0):.1%} "
      f"to {mode_counts[mode_fireplace]:.1%} of the entire dataset!")

# -----------------------------------------------------------
# 4. DIAGNOSTIC VISUALIZATION
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Garage Quality (Low missingness -> Mode is safe)
X_train["garage_qual"].value_counts(normalize=True).plot(
    kind="bar", ax=axes[0], color="steelblue", alpha=0.8, position=0, width=0.4, label="Original"
)
X_train["garage_qual"].fillna(X_train["garage_qual"].mode()[0]).value_counts(normalize=True).plot(
    kind="bar", ax=axes[0], color="darkorange", alpha=0.6, position=1, width=0.4, label="Mode Imputed"
)
axes[0].set_title("Garage Quality (4% Nulls): Mode Imputation Preserves Balance")
axes[0].set_ylabel("Proportion")
axes[0].legend()

# Plot 2: Fireplace Quality (48% Nulls -> Mode distorts distribution)
mode_counts.plot(
    kind="bar", ax=axes[1], color="crimson", alpha=0.7, position=0, width=0.4, label="Mode Imputed (Distorted)"
)
cat_counts.plot(
    kind="bar", ax=axes[1], color="seagreen", alpha=0.7, position=1, width=0.4, label="Explicit 'Missing' Category"
)
axes[1].set_title("Fireplace Quality (48% Nulls): Mode vs. 'Missing' Category")
axes[1].legend()

plt.tight_layout()
plt.savefig("visualizations/categorical_imputation_audit.png")
plt.close()
print("Saved comparison plot -> 'visualizations/categorical_imputation_audit.png'")

# -----------------------------------------------------------
# 5. SCIKIT-LEARN PIPELINE ORCHESTRATION
# -----------------------------------------------------------
print("\n--- 5. Orchestrating with Scikit-Learn Pipeline ---")

# Preprocessing strategy:
# - garage_qual (low nulls): SimpleImputer(strategy='most_frequent')
# - fireplace_qual (high nulls): SimpleImputer(strategy='constant', fill_value='Missing')
# Followed by OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ("garage_imp", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ]), ["garage_qual"]),
        
        ("fireplace_imp", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ]), ["fireplace_qual"])
    ],
    remainder="passthrough"  # gr_liv_area passes through directly
)

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0))
])

# Fit strictly on raw training DataFrame
full_pipeline.fit(X_train, y_train)

train_r2 = full_pipeline.score(X_train, y_train)
test_r2 = full_pipeline.score(X_test, y_test)

print(f"Pipeline fitted successfully.")
print(f"Training Set R²: {train_r2:.4f}")
print(f"Holdout Test R²: {test_r2:.4f}")

# -----------------------------------------------------------
# 6. INFERENCE AUDIT ON RAW UNPROCESSED PAYLOADS
# -----------------------------------------------------------
print("\n--- 6. Direct Inference on Unseen Records ---")

incoming_payloads = pd.DataFrame([
    {
        "gr_liv_area": 1850.0,
        "garage_qual": np.nan,         # Imputed with mode ('TA')
        "fireplace_qual": np.nan       # Imputed with 'Missing' category
    },
    {
        "gr_liv_area": 2200.0,
        "garage_qual": "Ex",
        "fireplace_qual": "Gd"         # Valid fireplace present
    }
])

predicted_prices = full_pipeline.predict(incoming_payloads)

for idx, price in enumerate(predicted_prices):
    print(f"House {idx + 1} Predicted Value: ${price:,.2f}")