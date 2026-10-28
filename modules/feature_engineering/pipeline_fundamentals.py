import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

print("=" * 65)
print("MACHINE LEARNING PIPELINES (PART 1: ARCHITECTURE & BASELINE)")
print("=" * 65)

# -----------------------------------------------------------
# 1. LOAD & CLEAN TITANIC DATASET
# -----------------------------------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Drop irrelevant high-cardinality identifiers
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
# Target: Survived
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Data Loaded: Train rows={X_train.shape[0]}, Test rows={X_test.shape[0]}")
print(f"Features: {list(X_train.columns)}")
print(f"Missing Values in Train:\n{X_train.isnull().sum()[X_train.isnull().sum() > 0]}")

# ===========================================================
# 2. APPROACH A: THE MANUAL WORKFLOW (WITHOUT PIPELINE)
# ===========================================================
print("\n--- Approach A: Manual Slicing & Concatenation ---")

# Step 1: Imputation
si_age = SimpleImputer(strategy="mean")
si_embarked = SimpleImputer(strategy="most_frequent")

train_age = si_age.fit_transform(X_train[["Age"]])
test_age = si_age.transform(X_test[["Age"]])

train_embarked = si_embarked.fit_transform(X_train[["Embarked"]])
test_embarked = si_embarked.transform(X_test[["Embarked"]])

# Step 2: One-Hot Encoding
ohe_sex = OneHotEncoder(drop="first", sparse_output=False)
ohe_emb = OneHotEncoder(drop="first", sparse_output=False)

train_sex = ohe_sex.fit_transform(X_train[["Sex"]])
test_sex = ohe_sex.transform(X_test[["Sex"]])

train_emb_ohe = ohe_emb.fit_transform(train_embarked)
test_emb_ohe = ohe_emb.transform(test_embarked)

# Step 3: Extract untreated numeric columns
untreated_cols = ["Pclass", "SibSp", "Parch", "Fare"]
train_untreated = X_train[untreated_cols].values
test_untreated = X_test[untreated_cols].values

# Step 4: Manual Concatenation (CRITICAL ORDER DEPENDENCE)
X_train_manual = np.concatenate(
    [train_untreated, train_age, train_sex, train_emb_ohe], axis=1
)
X_test_manual = np.concatenate(
    [test_untreated, test_age, test_sex, test_emb_ohe], axis=1
)

# Step 5: Fit Classifier
clf_manual = DecisionTreeClassifier(random_state=42)
clf_manual.fit(X_train_manual, y_train)
acc_manual = accuracy_score(y_test, clf_manual.predict(X_test_manual))
print(f"Manual Model Accuracy: {acc_manual * 100:.2f}%")
print(f"Manual Transformed Matrix Shape: {X_train_manual.shape}")

# ===========================================================
# 3. APPROACH B: UNIFIED PIPELINE (THE PRODUCTION WAY)
# ===========================================================
print("\n--- Approach B: Unified Scikit-Learn Pipeline ---")

# Step 1: Composite Preprocessor with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num_imputer", SimpleImputer(strategy="mean"), ["Age"]),
        ("cat_imputer_ohe", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ]), ["Embarked"]),
        ("sex_ohe", OneHotEncoder(drop="first", sparse_output=False), ["Sex"])
    ],
    remainder="passthrough"  # Retains Pclass, SibSp, Parch, Fare automatically
)

# Step 2: Construct the End-to-End Pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

# Step 3: Single .fit() call on RAW unprocessed DataFrames
pipeline.fit(X_train, y_train)

# Step 4: Single .predict() call on RAW test data
y_pred_pipe = pipeline.predict(X_test)
acc_pipe = accuracy_score(y_test, y_pred_pipe)

print(f"Pipeline Model Accuracy: {acc_pipe * 100:.2f}%")
print(f"Pipeline successfully encapsulated all steps: {list(pipeline.named_steps.keys())}")

# -----------------------------------------------------------
# 4. INFERENCE TEST: PASSING RAW REAL-WORLD RECORD
# -----------------------------------------------------------
print("\n--- 4. Direct Inference on Raw Passenger Record ---")

# Simulate a new passenger input directly from a frontend form / API request
raw_passenger = pd.DataFrame([{
    "Pclass": 2,
    "Sex": "female",
    "Age": np.nan,        # Missing age handled on the fly
    "SibSp": 1,
    "Parch": 0,
    "Fare": 32.50,
    "Embarked": "S"
}])

prediction = pipeline.predict(raw_passenger)[0]
outcome = "Survived" if prediction == 1 else "Deceased"
print(f"Input Record:\n{raw_passenger.to_dict(orient='records')[0]}")
print(f"Pipeline Predicted Outcome: {outcome} (Direct raw inference without manual transforms!)")