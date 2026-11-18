import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 72)
print("PCA PART 3 (STEP 2): OPTIMAL COMPONENT SELECTION & FAILURE MODES")
print("=" * 72)

# -----------------------------------------------------------
# 1. CUMULATIVE EXPLAINED VARIANCE ON MNIST (784 FEATURES)
# -----------------------------------------------------------
print("\nLoading & standardizing MNIST training partition...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = mnist.data[:42000].astype(np.float32)
y = mnist.target[:42000].astype(int)

X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Fit full PCA (all 784 components) to inspect eigenvalue spectrum
pca_full = PCA(n_components=None, random_state=42)
pca_full.fit(X_train_scaled)

cum_variance = np.cumsum(pca_full.explained_variance_ratio_)

k_90 = int(np.argmax(cum_variance >= 0.90) + 1)
k_95 = int(np.argmax(cum_variance >= 0.95) + 1)

# Verify exact parity with Scikit-Learn's built-in float variance selector
pca_auto_90 = PCA(n_components=0.90, random_state=42).fit(X_train_scaled)
assert pca_auto_90.n_components_ == k_90, "Cumulative variance threshold mismatch!"

print(f"\nPCA Components Matrix Shape: {pca_full.components_.shape}")
print(f"Top 5 Explained Variance Ratios: {np.round(pca_full.explained_variance_ratio_[:5] * 100, 2)}%")
print(f"Components needed for 90% Variance: {k_90} / 784 ({100 * (1 - k_90 / 784):.1f}% dimensionality reduction)")
print(f"Components needed for 95% Variance: {k_95} / 784 ({100 * (1 - k_95 / 784):.1f}% dimensionality reduction)")

# -----------------------------------------------------------
# 2. SIMULATE THE 3 GEOMETRIC FAILURE MODES OF LINEAR PCA
# -----------------------------------------------------------
np.random.seed(42)

# Failure Mode 1: Isotropic Spherical Cloud (Equal variance in all directions)
X_isotropic = np.random.normal(loc=0.0, scale=1.0, size=(250, 2))
pca_iso = PCA(n_components=2).fit(X_isotropic)

# Failure Mode 2: Parallel Elongated Classes (PC1 collapses classes together)
t_bar = np.linspace(-4, 4, 125)
noise_bar = np.random.normal(0, 0.25, size=125)
# Two parallel diagonal bars shifted perpendicularly
c0_x = t_bar - 1.2 + noise_bar
c0_y = t_bar + 1.2 + np.random.normal(0, 0.25, size=125)
c1_x = t_bar + 1.2 + np.random.normal(0, 0.25, size=125)
c1_y = t_bar - 1.2 + np.random.normal(0, 0.25, size=125)

X_parallel = np.vstack([np.column_stack([c0_x, c0_y]), np.column_stack([c1_x, c1_y])])
y_parallel = np.array([0] * 125 + [1] * 125)
pca_par = PCA(n_components=2).fit(X_parallel)
Z_par_1d = pca_par.transform(X_parallel)[:, 0]

# Failure Mode 3: Non-Linear Manifold (y = x^2 Parabola)
x_curve = np.linspace(-3, 3, 220)
y_curve = x_curve ** 2 + np.random.normal(0, 0.25, size=220)
X_nonlinear = np.column_stack([x_curve, y_curve])
X_nonlinear_centered = X_nonlinear - X_nonlinear.mean(axis=0)
pca_nonlin = PCA(n_components=1).fit(X_nonlinear_centered)
Z_nonlin_1d = pca_nonlin.transform(X_nonlinear_centered)
X_nonlin_reconstructed = pca_nonlin.inverse_transform(Z_nonlin_1d)

print("\n--- Failure Mode Diagnostics ---")
print(f"1. Isotropic Data Variance Split: PC1 = {pca_iso.explained_variance_ratio_[0]:.1%}, PC2 = {pca_iso.explained_variance_ratio_[1]:.1%} (No dominant axis)")
print(f"2. Parallel Classes PC1 Variance: {pca_par.explained_variance_ratio_[0]:.1%} captured, yet classes overlap 100% along PC1")
print(f"3. Non-Linear Parabola (y = x^2): 1D linear projection flattens curvature into a straight line")

# -----------------------------------------------------------
# 3. GENERATE 2x2 DIAGNOSTIC VISUALIZATION
# -----------------------------------------------------------
os.makedirs("visualizations", exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))

# Panel 1: MNIST Cumulative Explained Variance Curve
ax = axes[0, 0]
components_axis = np.arange(1, len(cum_variance) + 1)
ax.plot(components_axis, cum_variance * 100, color="navy", lw=2.3, label="Cumulative Explained Variance")
ax.axhline(90, color="darkorange", linestyle="--", lw=1.5, label=f"90% Threshold (k = {k_90})")
ax.axvline(k_90, color="darkorange", linestyle=":", lw=1.5)
ax.axhline(95, color="crimson", linestyle="--", lw=1.5, label=f"95% Threshold (k = {k_95})")
ax.axvline(k_95, color="crimson", linestyle=":", lw=1.5)
ax.scatter([k_90, k_95], [90, 95], color=["darkorange", "crimson"], s=65, zorder=5)
ax.set_title("1. MNIST Cumulative Explained Variance (Elbow Curve)")
ax.set_xlabel("Number of Principal Components (k)")
ax.set_ylabel("Cumulative Explained Variance (%)")
ax.set_xlim(0, 784)
ax.set_ylim(0, 102)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="lower right")

# Panel 2: Failure Mode 1 - Isotropic Spherical Distribution
ax = axes[0, 1]
ax.scatter(X_isotropic[:, 0], X_isotropic[:, 1], alpha=0.55, color="steelblue")
for idx, (vec, ratio, color) in enumerate(
    zip(pca_iso.components_, pca_iso.explained_variance_ratio_, ["crimson", "darkorange"])
):
    ax.quiver(0, 0, vec[0] * 2.2, vec[1] * 2.2, angles="xy", scale_units="xy", scale=1,
              color=color, width=0.009, label=f"PC{idx + 1} ({ratio:.1%} Var)")
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_title("2. Failure Mode 1: Isotropic Data (Equal Variance)")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper left")

# Panel 3: Failure Mode 2 - Unsupervised PC1 Collapses Distinct Classes
ax = axes[1, 0]
ax.scatter(X_parallel[y_parallel == 0, 0], X_parallel[y_parallel == 0, 1], color="royalblue", alpha=0.65, label="Class 0")
ax.scatter(X_parallel[y_parallel == 1, 0], X_parallel[y_parallel == 1, 1], color="crimson", alpha=0.65, label="Class 1")
pc1_par = pca_par.components_[0]
t_axis = np.linspace(-5.5, 5.5, 100)
ax.plot(t_axis * pc1_par[0], t_axis * pc1_par[1], color="black", linestyle="--", lw=2, label="PC1 (Max Variance Axis)")
ax.set_title("3. Failure Mode 2: PC1 Projection Merges Parallel Classes")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper left")

# Panel 4: Failure Mode 3 - Non-Linear Manifold Destroyed by Linear Projection
ax = axes[1, 1]
ax.scatter(X_nonlinear_centered[:, 0], X_nonlinear_centered[:, 1], color="teal", alpha=0.55, label="Original Curved Data (y = x²)")
ax.scatter(X_nonlin_reconstructed[:, 0], X_nonlin_reconstructed[:, 1], color="crimson", s=18, alpha=0.85, label="1D Linear PCA Projection")
ax.set_title("4. Failure Mode 3: Non-Linear Curve Crushed onto Line")
ax.set_xlabel("X (Centered)")
ax.set_ylabel("Y (Centered)")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper center")

plt.tight_layout()
plt.savefig("visualizations/pca_variance_and_failure_modes.png", dpi=150)
plt.close()
print("\nSaved variance selection & failure modes plot -> 'visualizations/pca_variance_and_failure_modes.png'")