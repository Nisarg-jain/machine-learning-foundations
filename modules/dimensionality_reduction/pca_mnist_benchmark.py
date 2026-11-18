import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

print("=" * 72)
print("PCA PART 3 (STEP 1): HIGH-DIMENSIONAL MNIST BENCHMARK & 2D/3D PROJECTION")
print("=" * 72)

# -----------------------------------------------------------
# 1. LOAD MNIST DATASET (42,000 IMAGES x 784 PIXELS)
# -----------------------------------------------------------
print("\nLoading MNIST dataset (784 pixel features)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = mnist.data[:42000].astype(np.float32)
y = mnist.target[:42000].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")

# -----------------------------------------------------------
# 2. BASELINE KNN ON RAW 784-DIMENSIONAL PIXEL SPACE
# -----------------------------------------------------------
t0 = time.perf_counter()
knn_baseline = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_baseline.fit(X_train, y_train)
y_pred_base = knn_baseline.predict(X_test)
base_time = time.perf_counter() - t0
base_acc = accuracy_score(y_test, y_pred_base)

print(f"\n[Baseline - 784 Features] Accuracy: {base_acc:.4%} | Fit+Predict Time: {base_time:.2f}s")

# -----------------------------------------------------------
# 3. ZERO-LEAKAGE STANDARDIZATION + PCA (100 COMPONENTS)
# -----------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca_100 = PCA(n_components=100, random_state=42)
X_train_pca100 = pca_100.fit_transform(X_train_scaled)
X_test_pca100 = pca_100.transform(X_test_scaled)

t0 = time.perf_counter()
knn_pca = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_pca.fit(X_train_pca100, y_train)
y_pred_pca100 = knn_pca.predict(X_test_pca100)
pca_time = time.perf_counter() - t0
pca_acc = accuracy_score(y_test, y_pred_pca100)

speedup = base_time / max(pca_time, 1e-6)
print(
    f"[PCA - 100 Components]   Accuracy: {pca_acc:.4%} | Fit+Predict Time: {pca_time:.2f}s "
    f"({speedup:.1f}x faster, 87.2% fewer features)"
)

# -----------------------------------------------------------
# 4. ACCURACY VS. NUMBER OF PRINCIPAL COMPONENTS SWEEP
# -----------------------------------------------------------
pca_full = PCA(n_components=400, random_state=42)
X_train_400 = pca_full.fit_transform(X_train_scaled)
X_test_400 = pca_full.transform(X_test_scaled)

component_grid = [1, 2, 3, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400]
sweep_accuracies = []

print("\nSweeping component counts to measure accuracy vs. dimensionality...")
for k in component_grid:
    clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    clf.fit(X_train_400[:, :k], y_train)
    acc_k = accuracy_score(y_test, clf.predict(X_test_400[:, :k]))
    sweep_accuracies.append(acc_k)
    print(f"  n_components = {k:3d} -> Test Accuracy: {acc_k:.4%}")

# -----------------------------------------------------------
# 5. GENERATE 3-PANEL BENCHMARK + 2D & 3D VISUALIZATIONS
# -----------------------------------------------------------
os.makedirs("visualizations", exist_ok=True)
fig = plt.figure(figsize=(18, 5.5))

# Panel 1: Accuracy vs. Number of Components
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(component_grid, [a * 100 for a in sweep_accuracies], marker="o", color="navy", lw=2)
ax1.axhline(base_acc * 100, color="crimson", linestyle="--", lw=1.6, label=f"784D Baseline ({base_acc:.2%})")
ax1.axvline(100, color="seagreen", linestyle=":", lw=1.6, label=f"100D PCA ({pca_acc:.2%})")
ax1.set_title("1. KNN Accuracy vs. Principal Components")
ax1.set_xlabel("Number of Principal Components (k)")
ax1.set_ylabel("Test Classification Accuracy (%)")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="lower right")

# Sample 3,000 test points for crisp 2D & 3D scatter rendering
vis_idx = np.arange(3000)
Z_vis = X_test_400[vis_idx, :3]
y_vis = y_test[vis_idx]

# Panel 2: 2D PCA Subspace (PC1 vs. PC2)
ax2 = fig.add_subplot(1, 3, 2)
scatter_2d = ax2.scatter(
    Z_vis[:, 0], Z_vis[:, 1], c=y_vis, cmap="tab10", s=10, alpha=0.7
)
ax2.set_title("2. MNIST Compressed to 2D (PC1 vs. PC2)")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
cbar2 = plt.colorbar(scatter_2d, ax=ax2, ticks=range(10))
cbar2.set_label("Digit Class")

# Panel 3: 3D PCA Subspace (PC1, PC2, PC3)
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
scatter_3d = ax3.scatter(
    Z_vis[:, 0], Z_vis[:, 1], Z_vis[:, 2], c=y_vis, cmap="tab10", s=9, alpha=0.65
)
ax3.set_title("3. MNIST Compressed to 3D (PC1, PC2, PC3)")
ax3.set_xlabel("PC1")
ax3.set_ylabel("PC2")
ax3.set_zlabel("PC3")

plt.tight_layout()
plt.savefig("visualizations/pca_mnist_benchmark_and_projection.png", dpi=150)
plt.close()
print("\nSaved MNIST benchmark & 2D/3D projection -> 'visualizations/pca_mnist_benchmark_and_projection.png'")