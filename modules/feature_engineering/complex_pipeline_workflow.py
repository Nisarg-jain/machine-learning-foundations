import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

print("=" * 65)
print("ADVANCED ML PIPELINES: CHAINING, SELECTION, CV & EXPORT")
print("=" * 65)

# -----------------------------------------------------------
# 1. LOAD RAW TITANIC DATASET
# -----------------------------------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Remove pure non-predictive identifiers
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Data Partitions: Train={X_train.shape[0]} rows, Test={X_test.shape[0]} rows")
print(f"Columns present in raw input: {list(X_train.columns)}")

# -----------------------------------------------------------
# 2. DESIGNING SUB-PIPELINES & COMPOSITE COLUMNTRANSFORMER
# -----------------------------------------------------------
# Numerical columns needing imputation + scaling
num_cols = ["Age", "Fare"]
# Categorical columns with missing values needing imputation + OHE
cat_impute_cols = ["Embarked"]
# Categorical columns with no missing values needing OHE
cat_clean_cols = ["Sex"]

num_subpipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

cat_impute_subpipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
])

cat_clean_subpipeline = Pipeline([
    ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_subpipeline, num_cols),
        ("cat_impute", cat_impute_subpipeline, cat_impute_cols),
        ("cat_clean", cat_clean_subpipeline, cat_clean_cols)
    ],
    remainder="passthrough"  # Pclass, SibSp, Parch pass through untouched
)

# -----------------------------------------------------------
# 3. CONSTRUCTING FULL END-TO-END PIPELINE
# -----------------------------------------------------------
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selector", SelectKBest(score_func=chi2, k=6)),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

# -----------------------------------------------------------
# 4. CROSS-VALIDATION AUDIT (LEAKAGE-FREE EVALUATION)
# -----------------------------------------------------------
print("\n--- 4. Evaluating Pipeline via 5-Fold Cross-Validation ---")
cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring="accuracy")
print(f"5-Fold CV Accuracy Scores: {[round(s, 4) for s in cv_scores]}")
print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}% (Std: {cv_scores.std():.4f})")

# -----------------------------------------------------------
# 5. HYPERPARAMETER TUNING VIA GRIDSEARCHCV
# -----------------------------------------------------------
print("\n--- 5. Hyperparameter Tuning with Nested Step Syntax ---")

param_grid = {
    # Tune selector's k best features
    "feature_selector__k": [4, 6, "all"],
    # Tune tree complexity
    "classifier__max_depth": [3, 5, 8, None],
    "classifier__criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Optimal Hyperparameters Found:")
for param, val in grid_search.best_params_.items():
    print(f"  {param}: {val}")
print(f"Best GridSearch CV Score: {grid_search.best_score_ * 100:.2f}%")

best_pipeline = grid_search.best_estimator_

# -----------------------------------------------------------
# 6. INTERNAL PIPELINE INSPECTION & POST-MORTEM
# -----------------------------------------------------------
print("\n--- 6. Inspecting Internal Pipeline Parameters ---")
# Extract internal steps via named_steps dictionary
fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
fitted_num_imputer = fitted_preprocessor.named_transformers_["num"].named_steps["imputer"]
print(f"Learned Median Imputation values (Age, Fare): {fitted_num_imputer.statistics_}")

fitted_selector = best_pipeline.named_steps["feature_selector"]
print(f"Feature Selector Chi2 Scores: {fitted_selector.scores_.round(2)}")

# Evaluate on held-out test data
test_preds = best_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)
print(f"\nFinal Generalization Accuracy on Unseen Test Set: {test_accuracy * 100:.2f}%")

# -----------------------------------------------------------
# 7. SERIALIZATION: EXPORTING SINGLE PRODUCTION ARTIFACT
# -----------------------------------------------------------
artifact_path = "artifacts/titanic_pipeline.joblib"
joblib.dump(best_pipeline, artifact_path)
print(f"\nSuccessfully serialized complete pipeline artifact -> '{artifact_path}'")

# -----------------------------------------------------------
# 8. PRODUCTION SIMULATION: INFERENCE ON RAW JSON PAYLOAD
# -----------------------------------------------------------
print("\n--- 8. Simulating Production Web Service (Direct Raw Inference) ---")

# Load the single unified artifact
deployed_pipeline = joblib.load(artifact_path)

# Incoming payload from an external client (no preprocessing performed on caller side)
production_payload = pd.DataFrame([
    {
        "Pclass": 3,
        "Sex": "male",
        "Age": np.nan,        # Missing age handled automatically by pipeline imputer
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.89,
        "Embarked": "S"
    },
    {
        "Pclass": 1,
        "Sex": "female",
        "Age": 29.0,
        "SibSp": 1,
        "Parch": 2,
        "Fare": 151.55,
        "Embarked": "C"
    }
])

predictions = deployed_pipeline.predict(production_payload)
probabilities = deployed_pipeline.predict_proba(production_payload)

for idx, pred in enumerate(predictions):
    status = "Survived" if pred == 1 else "Deceased"
    surv_prob = probabilities[idx][1]
    print(f"Passenger {idx + 1}: Predicted Status = {status} (Survival Probability: {surv_prob:.2%})")