import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

print("=" * 65)
print("FEATURE ENGINEERING: FEATURE CONSTRUCTION & FEATURE SPLITTING")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC TITANIC-STYLE DATASET WITH UNTIDY STRINGS
# -----------------------------------------------------------
np.random.seed(42)
n_samples = 900

pclass = np.random.choice([1, 2, 3], size=n_samples, p=[0.24, 0.21, 0.55])
sibsp = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.68, 0.23, 0.04, 0.03, 0.02])
parch = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.76, 0.13, 0.09, 0.02])
age = np.random.normal(loc=29.5, scale=13.0, size=n_samples).clip(1, 75)

# Inject ~15% missingness into Age
age[np.random.rand(n_samples) < 0.15] = np.nan

# Construct untidy Name strings packing Surname + Salutation + Given Name
surnames = ["Braund", "Cumings", "Heikkinen", "Futrelle", "Allen", "Moran", "McCarthy", "Palsson"]
given_male = ["Owen Harris", "William Henry", "James", "Timothy J", "John"]
given_female = ["John Bradley", "Laina", "Jacques Heath", "Elizabeth", "Margaret"]

names = []
sexes = []
titles_true = []

for i in range(n_samples):
    sur = np.random.choice(surnames)
    if np.random.rand() < 0.64:
        sex = "male"
        if not np.isnan(age[i]) and age[i] < 13:
            title = "Master"
        elif np.random.rand() < 0.04:
            title = "Dr"
        else:
            title = "Mr"
        given = np.random.choice(given_male)
    else:
        sex = "female"
        if np.random.rand() < 0.45:
            title = "Mrs"
        else:
            title = "Miss"
        given = np.random.choice(given_female)
    names.append(f"{sur}, {title}. {given}")
    sexes.append(sex)
    titles_true.append(title)

# True survival signal depends non-linearly on family size & extracted title
family_size_true = sibsp + parch + 1
small_family_bonus = ((family_size_true >= 2) & (family_size_true <= 4)).astype(int) * 0.85
large_family_penalty = (family_size_true > 4).astype(int) * -1.4
title_bonus = np.where(np.isin(titles_true, ["Mrs", "Miss", "Master"]), 1.35, -0.4)

logits = -0.8 - 0.65 * (pclass - 1) + small_family_bonus + large_family_penalty + title_bonus
probs = 1 / (1 + np.exp(-logits))
survived = np.random.binomial(n=1, p=probs, size=n_samples)

df = pd.DataFrame({
    "Pclass": pclass,
    "Sex": sexes,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Name": names,
    "Survived": survived
})

print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns.")
print("\nSample Raw Untidy Records:")
print(df[["Name", "SibSp", "Parch", "Survived"]].head(4))

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 2. VECTORIZED FEATURE CONSTRUCTION & SPLITTING FUNCTIONS
# -----------------------------------------------------------
def construct_family_type(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines SibSp + Parch + 1 into FamilySize, then discretizes into:
      0 -> Alone (size == 1)
      1 -> Small Family (2 <= size <= 4)
      2 -> Large Family (size >= 5)
    """
    family_size = X_df["SibSp"] + X_df["Parch"] + 1
    family_type = np.select(
        [family_size == 1, (family_size >= 2) & (family_size <= 4)],
        [0, 1],
        default=2
    )
    return pd.DataFrame({"family_type": family_type}, index=X_df.index)


def split_name_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits untidy 'Surname, Title. GivenName' strings into atomic features:
      - title: Extracted salutation (Mr, Mrs, Miss, Master, Dr, etc.)
      - is_married: Binary indicator (1 if title == 'Mrs', else 0)
    """
    titles = (
        X_df["Name"]
        .str.split(",", expand=True)[1]
        .str.split(".", expand=True)[0]
        .str.strip()
    )
    is_married = (titles == "Mrs").astype(int)
    return pd.DataFrame({"title": titles, "is_married": is_married}, index=X_df.index)

# -----------------------------------------------------------
# 3. DIAGNOSTIC AUDIT & VISUALIZATION
# -----------------------------------------------------------
os.makedirs("visualizations", exist_ok=True)

audit_df = X_train.copy()
audit_df["Survived"] = y_train
audit_df["FamilyType"] = construct_family_type(X_train)["family_type"].map({
    0: "Alone (1)",
    1: "Small (2-4)",
    2: "Large (5+)"
})
audit_df["Title"] = split_name_features(X_train)["title"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

audit_df.groupby("FamilyType")["Survived"].mean().reindex(["Alone (1)", "Small (2-4)", "Large (5+)"]).plot(
    kind="bar", ax=axes[0], color=["slategray", "seagreen", "crimson"], edgecolor="black"
)
axes[0].set_title("Constructed Feature: Survival Rate by Family Type")
axes[0].set_ylabel("Survival Probability")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis="x", rotation=0)

audit_df.groupby("Title")["Survived"].mean().sort_values(ascending=False).plot(
    kind="bar", ax=axes[1], color="steelblue", edgecolor="black"
)
axes[1].set_title("Split Feature: Survival Rate by Extracted Salutation")
axes[1].set_ylabel("Survival Probability")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("visualizations/feature_construction_audit.png")
plt.close()
print("\nSaved diagnostic chart -> 'visualizations/feature_construction_audit.png'")

# -----------------------------------------------------------
# 4. BENCHMARKING: RAW BASELINE VS. ENGINEERED PIPELINE
# -----------------------------------------------------------
print("\n--- 4. 5-Fold Cross-Validation Benchmark ---")

# Pipeline A: Baseline (Raw columns only, no construction or string splitting)
baseline_preprocessor = ColumnTransformer(
    transformers=[
        ("age_imp", SimpleImputer(strategy="median"), ["Age"]),
        ("sex_ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), ["Sex"]),
        ("raw_num", "passthrough", ["Pclass", "SibSp", "Parch"])
    ]
)

pipe_baseline = Pipeline([
    ("prep", baseline_preprocessor),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, random_state=42))
])

# Pipeline B: Full Feature Construction + Feature Splitting
family_branch = Pipeline([
    ("construct", FunctionTransformer(construct_family_type)),
    ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
])

name_branch = Pipeline([
    ("split", FunctionTransformer(split_name_features)),
    ("encode", ColumnTransformer(
        transformers=[
            ("title_ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), ["title"])
        ],
        remainder="passthrough"  # Preserves binary 'is_married' column
    ))
])

engineered_preprocessor = ColumnTransformer(
    transformers=[
        ("age_imp", SimpleImputer(strategy="median"), ["Age"]),
        ("sex_ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), ["Sex"]),
        ("pclass_pass", "passthrough", ["Pclass"]),
        ("family_ctor", family_branch, ["SibSp", "Parch"]),
        ("name_split", name_branch, ["Name"])
    ]
)

pipe_engineered = Pipeline([
    ("prep", engineered_preprocessor),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, random_state=42))
])

cv_base = cross_val_score(pipe_baseline, X_train, y_train, cv=5, scoring="accuracy").mean()
cv_eng = cross_val_score(pipe_engineered, X_train, y_train, cv=5, scoring="accuracy").mean()

pipe_baseline.fit(X_train, y_train)
pipe_engineered.fit(X_train, y_train)

print(f"Baseline Pipeline   -> 5-Fold CV: {cv_base * 100:.2f}% | Holdout Test: {pipe_baseline.score(X_test, y_test) * 100:.2f}%")
print(f"Engineered Pipeline -> 5-Fold CV: {cv_eng * 100:.2f}% | Holdout Test: {pipe_engineered.score(X_test, y_test) * 100:.2f}%")
print(f"Cross-Validation Lift: +{(cv_eng - cv_base) * 100:.2f}%")

# -----------------------------------------------------------
# 5. DIRECT INFERENCE ON RAW UNPROCESSED PAYLOAD
# -----------------------------------------------------------
print("\n--- 5. Inference on Raw Unprocessed Passenger Payload ---")

raw_passenger = pd.DataFrame([{
    "Pclass": 1,
    "Sex": "female",
    "Age": np.nan,
    "SibSp": 1,
    "Parch": 1,
    "Name": "Cumings, Mrs. John Bradley"
}])

pred = pipe_engineered.predict(raw_passenger)[0]
prob = pipe_engineered.predict_proba(raw_passenger)[0][1]

status = "Survived" if pred == 1 else "Did Not Survive"
print(f"Incoming Record: {raw_passenger.to_dict(orient='records')[0]}")
print(f"Prediction: {status} (Survival Probability: {prob:.2%})")