import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("=" * 65)
print("FEATURE SCALING: STANDARDIZATION & ALGORITHM BENCHMARK")
print("=" * 65)

# Synthetic social network ad purchase dataset (mimics Social_Network_Ads.csv)
np.random.seed(42)
n_samples = 400

# Age: 18 - 60 years
age = np.random.randint(18, 60, size=n_samples)
# Estimated Salary: $15,000 - $150,000
salary = np.random.randint(15000, 150000, size=n_samples)

# Non-linear probability of purchase based on age & salary
logits = (age - 35) * 0.12 + (salary - 75000) * 0.00004
probs = 1 / (1 + np.exp(-logits))
purchased = (probs > 0.5).astype(int)

df = pd.DataFrame({
    "Age": age,
    "EstimatedSalary": salary,
    "Purchased": purchased
})

print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head(4))

# -----------------------------------------------------------
# 1. TRAIN / TEST SPLIT (PREVENT DATA LEAKAGE)
# -----------------------------------------------------------
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------------------------------------
# 2. FIT STANDARDIZATION ON TRAIN ONLY
# -----------------------------------------------------------
scaler = StandardScaler()
# Fit calculates mean and std on training partition
X_train_scaled = scaler.fit_transform(X_train)
# Transform applies learned mean and std to test partition
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for statistical validation
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

print("\n--- 2. STATISTICAL VALIDATION OF SCALED FEATURES ---")
print("Original X_train Stats:")
print(f"  Age:    Mean = {X_train['Age'].mean():.2f}, Std = {X_train['Age'].std():.2f}")
print(f"  Salary: Mean = {X_train['EstimatedSalary'].mean():.2f}, Std = {X_train['EstimatedSalary'].std():.2f}")

print("\nScaled X_train Stats (Expect Mean ≈ 0, Std ≈ 1):")
print(f"  Age:    Mean = {X_train_scaled_df['Age'].mean():.4f}, Std = {X_train_scaled_df['Age'].std():.4f}")
print(f"  Salary: Mean = {X_train_scaled_df['EstimatedSalary'].mean():.4f}, Std = {X_train_scaled_df['EstimatedSalary'].std():.4f}")

# -----------------------------------------------------------
# 3. EMPIRICAL MODEL PERFORMANCE BENCHMARK
# -----------------------------------------------------------
print("\n--- 3. ALGORITHM SENSITIVITY BENCHMARK ---")

# A. Logistic Regression (Distance / Gradient Descent Based)
clf_unscaled = LogisticRegression(random_state=42)
clf_unscaled.fit(X_train, y_train)
acc_unscaled_lr = accuracy_score(y_test, clf_unscaled.predict(X_test))

clf_scaled = LogisticRegression(random_state=42)
clf_scaled.fit(X_train_scaled, y_train)
acc_scaled_lr = accuracy_score(y_test, clf_scaled.predict(X_test_scaled))

# B. Decision Tree (Tree Partitioning - Rule Based)
tree_unscaled = DecisionTreeClassifier(random_state=42)
tree_unscaled.fit(X_train, y_train)
acc_unscaled_dt = accuracy_score(y_test, tree_unscaled.predict(X_test))

tree_scaled = DecisionTreeClassifier(random_state=42)
tree_scaled.fit(X_train_scaled, y_train)
acc_scaled_dt = accuracy_score(y_test, tree_scaled.predict(X_test_scaled))

results_df = pd.DataFrame({
    "Algorithm": ["Logistic Regression", "Decision Tree Classifier"],
    "Unscaled Accuracy": [f"{acc_unscaled_lr * 100:.2f}%", f"{acc_unscaled_dt * 100:.2f}%"],
    "Scaled Accuracy": [f"{acc_scaled_lr * 100:.2f}%", f"{acc_scaled_dt * 100:.2f}%"],
    "Scale Sensitive?": ["YES (Gradients impacted)", "NO (Invariant to scale)"]
})
print(results_df.to_string(index=False))

# -----------------------------------------------------------
# 4. OUTLIER BEHAVIOR IN STANDARDIZATION
# -----------------------------------------------------------
print("\n--- 4. OUTLIER PERSISTENCE DEMONSTRATION ---")
extreme_sample = pd.DataFrame({
    "Age": [105],              # Extreme outlier age
    "EstimatedSalary": [500000] # Extreme outlier salary
})
extreme_scaled = scaler.transform(extreme_sample)
print(f"Original Outlier Sample: Age=105, Salary=$500,000")
print(f"Standardized Outlier Z-Scores: Age Z={extreme_scaled[0, 0]:.2f}, Salary Z={extreme_scaled[0, 1]:.2f}")
print("-> Note: Standardization retains extreme outliers; Z-scores far exceed [-3, +3].")