import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer, Binarizer
from sklearn.tree import DecisionTreeClassifier

print("=" * 65)
print("FEATURE ENGINEERING: BINNING (DISCRETIZATION) & BINARIZATION")
print("=" * 65)

# -----------------------------------------------------------
# 1. LOAD TITANIC BENCHMARK DATASET
# -----------------------------------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Retain continuous features, familial counts, and target
cols = ["Age", "Fare", "SibSp", "Parch", "Survived"]
df = df[cols].dropna().copy()

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Data ready: Train rows={X_train.shape[0]}, Test rows={X_test.shape[0]}")
print(f"Features: {list(X_train.columns)}")

# -----------------------------------------------------------
# 2. COMPARATIVE BINNING STRATEGIES (KBinsDiscretizer)
# -----------------------------------------------------------
print("\n--- 2. Comparing Discretization Strategies on 'Age' & 'Fare' ---")

n_bins = 5
strategies = ["uniform", "quantile", "kmeans"]
bin_results = {}

for strat in strategies:
    # encode='ordinal' produces clean integer bins: 0, 1, ..., n_bins-1
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strat, random_state=42)
    
    # Fit strictly on training partition
    train_binned = kbd.fit_transform(X_train[["Age", "Fare"]])
    test_binned = kbd.transform(X_test[["Age", "Fare"]])
    
    # Evaluate a baseline DecisionTree on binned features
    cv_score = cross_val_score(
        DecisionTreeClassifier(random_state=42), 
        train_binned, 
        y_train, 
        cv=5, 
        scoring="accuracy"
    ).mean()
    
    bin_results[strat] = {
        "transformer": kbd,
        "train_binned": train_binned,
        "cv_score": cv_score
    }
    
    print(f"Strategy: {strat:<9} | 5-Fold CV Accuracy: {cv_score * 100:.2f}%")
    print(f"  Age Bin Edges:  {[round(x, 1) for x in kbd.bin_edges_[0]]}")
    print(f"  Fare Bin Edges: {[round(x, 1) for x in kbd.bin_edges_[1]]}")

# -----------------------------------------------------------
# 3. DOMAIN-SPECIFIC CUSTOM BINNING (pd.cut)
# -----------------------------------------------------------
print("\n--- 3. Domain Knowledge Binning via pd.cut ---")

# Real-world age brackets based on demographics
age_cutoffs = [0, 12, 18, 60, 100]
age_labels = ["Child", "Adolescent", "Adult", "Senior"]

custom_age_binned = pd.cut(
    X_train["Age"], 
    bins=age_cutoffs, 
    labels=age_labels, 
    right=True
)

print("Custom Age Demographic Distribution:")
print(custom_age_binned.value_counts().sort_index())

# -----------------------------------------------------------
# 4. BINARIZATION (THRESHOLDING VIA Binarizer)
# -----------------------------------------------------------
print("\n--- 4. Feature Binarization (Solo Traveler Indicator) ---")

# Step A: Synthesize family size = SibSp + Parch
family_train = (X_train["SibSp"] + X_train["Parch"]).values.reshape(-1, 1)

# Step B: Threshold at 0 -> Any family > 0 becomes 1 (Accompanied), exactly 0 becomes 0 (Solo)
binarizer = Binarizer(threshold=0.0)
is_accompanied = binarizer.fit_transform(family_train)

# Invert to create an intuitive 'is_alone' binary flag (1 = Alone, 0 = With Family)
is_alone_train = (is_accompanied == 0).astype(int)

sample_audit = pd.DataFrame({
    "SibSp": X_train["SibSp"].values[:6],
    "Parch": X_train["Parch"].values[:6],
    "Raw_Family_Size": family_train.ravel()[:6],
    "Is_Alone_Flag": is_alone_train.ravel()[:6]
})
print("Binarizer Application Sample:")
print(sample_audit.to_string(index=False))

# -----------------------------------------------------------
# 5. VISUALIZING DISTRIBUTION RESHAPING
# -----------------------------------------------------------
print("\n--- 5. Generating Diagnostic Distribution Plots ---")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Raw Fare Distribution
axes[0, 0].hist(X_train["Fare"], bins=30, color="crimson", edgecolor="black", alpha=0.7)
axes[0, 0].set_title("Original Continuous Fare (Heavy Right Skew)")
axes[0, 0].set_xlabel("Fare ($)")
axes[0, 0].set_ylabel("Count")

# Equal Width (Uniform) Binned Fare
axes[0, 1].hist(bin_results["uniform"]["train_binned"][:, 1], bins=n_bins, color="coral", edgecolor="black", rwidth=0.8)
axes[0, 1].set_title("Uniform Binning: Skew remains in first bin")
axes[0, 1].set_xlabel("Bin Index (0 to 4)")

# Equal Frequency (Quantile) Binned Fare
axes[1, 0].hist(bin_results["quantile"]["train_binned"][:, 1], bins=n_bins, color="seagreen", edgecolor="black", rwidth=0.8)
axes[1, 0].set_title("Quantile Binning: Flattens into equal counts per bin")
axes[1, 0].set_xlabel("Bin Index (0 to 4)")

# K-Means Binned Fare
axes[1, 1].hist(bin_results["kmeans"]["train_binned"][:, 1], bins=n_bins, color="royalblue", edgecolor="black", rwidth=0.8)
axes[1, 1].set_title("K-Means Binning: Adapts to 1D data clusters")
axes[1, 1].set_xlabel("Bin Index (0 to 4)")

plt.tight_layout()
plt.savefig("visualizations/binning_distributions_comparison.png")
plt.close()
print("Saved comparison figure -> 'visualizations/binning_distributions_comparison.png'")