import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

print("=" * 65)
print("FEATURE ENGINEERING: MULTIVARIATE KNN IMPUTATION")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC BENCHMARK DATASET (TITANIC ARCHITECTURE)
# -----------------------------------------------------------
np.random.seed(42)
n_samples = 1000

# Pclass: 1, 2, or 3
pclass = np.random.choice([1, 2, 3], size=n_samples, p=[0.25, 0.25, 0.50])

# Fare is heavily correlated with Pclass
fare = np.where(pclass == 1, np.random.exponential(55, n_samples) + 50,
        np.where(pclass == 2, np.random.exponential(20, n_samples) + 15,
                              np.random.exponential(8, n_samples) + 7))

# Age is correlated with Pclass (1st class passengers skew older)
base_age = np.where(pclass == 1, np.random.normal(38, 9, n_samples),
           np.where(pclass == 2, np.random.normal(29, 8, n_samples),
                                 np.random.normal(23, 7, n_samples))).clip(1, 75)

# Inject 20% missingness into Age
mask_age = np.random.rand(n_samples) < 0.20
age_missing = base_age.copy()
age_missing[mask_age] = np.nan

# Target: Survival probability
logits = -1.3 - (0.8 * (pclass - 1)) + (0.015 * fare) + (0.02 * base_age)
probs = 1 / (1 + np.exp(-logits))
survived = np.random.binomial(n=1, p=probs, size=n_samples)

df = pd.DataFrame({
    "pclass": pclass,
    "fare": fare,
    "age": age_missing,
    "survived": survived
})

print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
print(f"Missing Values in 'age': {df['age'].isna().sum()} ({df['age'].isna().mean():.1%})")

# Leakage-Free Train-Test Split
X = df[["pclass", "fare", "age"]]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 2. BASELINE BENCHMARK: UNIVARIATE MEDIAN IMPUTATION
# -----------------------------------------------------------
print("\n--- 2. Baseline: Univariate Median Imputation ---")

pipe_median = Pipeline([
    ("scaler", StandardScaler()),
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", LogisticRegression(random_state=42))
])

median_cv = cross_val_score(pipe_median, X_train, y_train, cv=5, scoring="accuracy").mean()
print(f"SimpleImputer (Median) 5-Fold CV Accuracy: {median_cv * 100:.2f}%")

# -----------------------------------------------------------
# 3. BENCHMARKING KNN IMPUTER: TUNING K AND WEIGHTS
# -----------------------------------------------------------
print("\n--- 3. Tuning KNNImputer (Neighbors K & Weight Functions) ---")

neighbor_values = [2, 3, 5, 8, 12]
weight_schemes = ["uniform", "distance"]

best_cv_score = 0.0
best_config = {}

for weights in weight_schemes:
    print(f"\nEvaluating weights='{weights}':")
    for k in neighbor_values:
        # Step: Scale -> Impute -> Model
        pipe_knn = Pipeline([
            ("scaler", StandardScaler()),
            ("imputer", KNNImputer(n_neighbors=k, weights=weights)),
            ("classifier", LogisticRegression(random_state=42))
        ])
        
        cv_score = cross_val_score(pipe_knn, X_train, y_train, cv=5, scoring="accuracy").mean()
        print(f"  k={k:<2} | 5-Fold CV Accuracy: {cv_score * 100:.2f}%")
        
        if cv_score > best_cv_score:
            best_cv_score = cv_score
            best_config = {"k": k, "weights": weights}

print(f"\nOptimal Setup: k={best_config['k']}, weights='{best_config['weights']}' "
      f"(CV Accuracy: {best_cv_score * 100:.2f}%)")

# -----------------------------------------------------------
# 4. FINAL PIPELINE EVALUATION ON HOLDOUT SET
# -----------------------------------------------------------
print("\n--- 4. Final Evaluation on Holdout Test Partition ---")

final_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn_imputer", KNNImputer(n_neighbors=best_config["k"], weights=best_config["weights"])),
    ("classifier", LogisticRegression(random_state=42))
])

# Fit strictly on training partition
final_pipeline.fit(X_train, y_train)

test_acc = final_pipeline.score(X_test, y_test)
print(f"Holdout Generalization Accuracy: {test_acc * 100:.2f}%")

# -----------------------------------------------------------
# 5. INFERENCE ON RAW USER RECORD
# -----------------------------------------------------------
print("\n--- 5. Direct Inference on Unprocessed User Record ---")

# Passenger: 1st class, high fare, but age is missing
raw_record = pd.DataFrame([{
    "pclass": 1,
    "fare": 135.0,
    "age": np.nan  # KNN borrows age from other high-fare, 1st-class passengers
}])

prediction = final_pipeline.predict(raw_record)[0]
prob = final_pipeline.predict_proba(raw_record)[0][1]

status = "Survived" if prediction == 1 else "Deceased"
print(f"Incoming Record: {raw_record.to_dict(orient='records')[0]}")
print(f"Model Prediction: {status} (Survival Probability: {prob:.2%})")