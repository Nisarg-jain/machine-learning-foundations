import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

print("=" * 65)
print("FEATURE ENGINEERING: CURSE OF DIMENSIONALITY SIMULATION")
print("=" * 65)

# -----------------------------------------------------------
# 1. EXPERIMENT A: DISTANCE CONCENTRATION IN HIGH DIMENSIONS
# -----------------------------------------------------------
print("\n--- 1. Auditing Distance Concentration (D_max vs. D_min) ---")

np.random.seed(42)
n_points = 300
dimensions_list = [2, 5, 10, 25, 50, 100, 200]
contrast_ratios = []

for d in dimensions_list:
    # Uniformly distributed points inside a d-dimensional unit hypercube [0, 1]^d
    points = np.random.uniform(0, 1, size=(n_points, d))
    dist_matrix = pairwise_distances(points, metric="euclidean")
    
    # Exclude self-distances along diagonal
    np.fill_diagonal(dist_matrix, np.nan)
    
    d_min = np.nanmin(dist_matrix, axis=1)
    d_max = np.nanmax(dist_matrix, axis=1)
    
    # Relative contrast ratio: (D_max - D_min) / D_min
    ratio = np.mean((d_max - d_min) / d_min)
    contrast_ratios.append(ratio)
    print(f"Dimensions: {d:<3} | Avg (D_max - D_min) / D_min Ratio: {ratio:.4f}")

# -----------------------------------------------------------
# 2. EXPERIMENT B: HUGHES PHENOMENON (ADDING NOISE FEATURES)
# -----------------------------------------------------------
print("\n--- 2. Simulating the Hughes Phenomenon (Accuracy vs. Dimensions) ---")

# Generate dataset with 5 true informative features and 150 pure noise features
X_full, y_full = make_classification(
    n_samples=400,
    n_features=155,
    n_informative=5,
    n_redundant=0,
    n_repeated=0,
    shuffle=False,  # First 5 columns are true signal; columns 5..154 are noise
    random_state=42
)

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.25, random_state=42
)

feature_counts = [1, 2, 3, 5, 10, 20, 40, 75, 115, 155]
knn_accuracies = []
lr_accuracies = []
fit_times_ms = []

for k_feats in feature_counts:
    X_tr_sub = X_train_full[:, :k_feats]
    X_te_sub = X_test_full[:, :k_feats]
    
    # Scale features
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_sub)
    X_te_scaled = scaler.transform(X_te_sub)
    
    # Evaluate KNN
    t0 = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr_scaled, y_train)
    knn_acc = knn.score(X_te_scaled, y_test)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    # Evaluate Logistic Regression
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_tr_scaled, y_train)
    lr_acc = lr.score(X_te_scaled, y_test)
    
    knn_accuracies.append(knn_acc)
    lr_accuracies.append(lr_acc)
    fit_times_ms.append(elapsed_ms)
    
    print(f"Features: {k_feats:<3} | KNN Test Acc: {knn_acc * 100:5.2f}% | LogReg Test Acc: {lr_acc * 100:5.2f}%")

# -----------------------------------------------------------
# 3. DIAGNOSTIC VISUALIZATION
# -----------------------------------------------------------
os.makedirs("visualizations", exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# Plot 1: Distance Concentration Collapse
axes[0].plot(dimensions_list, contrast_ratios, marker="o", color="crimson", lw=2.2)
axes[0].set_title("1. Distance Concentration Collapse")
axes[0].set_xlabel("Number of Dimensions (d)")
axes[0].set_ylabel("(D_max - D_min) / D_min")
axes[0].grid(True, linestyle=":", alpha=0.6)

# Plot 2: Hughes Phenomenon Curve
axes[1].plot(feature_counts, [a * 100 for a in knn_accuracies], marker="o", color="navy", lw=2, label="KNN (k=5)")
axes[1].plot(feature_counts, [a * 100 for a in lr_accuracies], marker="s", color="seagreen", lw=2, label="Logistic Regression")
axes[1].axvline(5, color="black", linestyle="--", label="Optimal Signal Dim (d=5)")
axes[1].set_title("2. Hughes Phenomenon: Accuracy Collapse")
axes[1].set_xlabel("Total Features (5 Signal + Noise)")
axes[1].set_ylabel("Holdout Test Accuracy (%)")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.6)

# Plot 3: Compute Overhead Growth
axes[2].plot(feature_counts, fit_times_ms, marker="^", color="darkorange", lw=2)
axes[2].set_title("3. Compute Latency Growth")
axes[2].set_xlabel("Total Features (d)")
axes[2].set_ylabel("Fit + Predict Latency (ms)")
axes[2].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("visualizations/curse_of_dimensionality_diagnostics.png")
plt.close()
print("\nSaved diagnostic plots -> 'visualizations/curse_of_dimensionality_diagnostics.png'")

# -----------------------------------------------------------
# 4. PREVIEW: COMBATING HIGH DIMENSIONS WITH PCA PIPELINE
# -----------------------------------------------------------
print("\n--- 4. Dimensionality Reduction Recovery via PCA Pipeline ---")

# Raw 155-dimensional noisy input
pipe_raw_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
pipe_raw_knn.fit(X_train_full, y_train)
raw_acc = pipe_raw_knn.score(X_test_full, y_test)

# Compressed to 5 principal components via Feature Extraction (PCA)
pipe_pca_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=5, random_state=42)),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
pipe_pca_knn.fit(X_train_full, y_train)
pca_acc = pipe_pca_knn.score(X_test_full, y_test)

print(f"KNN on Raw 155 Features:      {raw_acc * 100:.2f}%")
print(f"KNN with PCA (5 Components):  {pca_acc * 100:.2f}% (Recovered +{((pca_acc - raw_acc) * 100):.2f}% accuracy)")