import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("=" * 68)
print("DIMENSIONALITY REDUCTION: PCA GEOMETRIC INTUITION & ROTATION")
print("=" * 68)

# -----------------------------------------------------------
# 1. SYNTHETIC CORRELATED DATASET (ROOMS VS. WASHROOMS)
# -----------------------------------------------------------
np.random.seed(42)
n_flats = 120

# Base flat size drives both Rooms and Washrooms linearly + minor noise
latent_size = np.random.normal(loc=0.0, scale=1.0, size=n_flats)
rooms = 3.0 + 1.1 * latent_size + np.random.normal(0, 0.22, size=n_flats)
washrooms = 2.5 + 1.05 * latent_size + np.random.normal(0, 0.22, size=n_flats)

# Place two specific reference points at opposite ends of the diagonal
# (Replicating the Red vs. Green distance preservation audit)
rooms[0], washrooms[0] = 1.2, 1.0   # Point A (Red - Small Flat)
rooms[1], washrooms[1] = 4.8, 4.5   # Point B (Green - Large Flat)

df = pd.DataFrame({"rooms": rooms, "washrooms": washrooms})

# Center the data at origin (0, 0) so rotation happens cleanly around the centroid
scaler = StandardScaler(with_std=False)
X_centered = scaler.fit_transform(df)

# -----------------------------------------------------------
# 2. FIT PCA & VERIFY ORTHOGONALITY + VARIANCE CONSERVATION
# -----------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_centered)

pc1_vec = pca.components_[0]
pc2_vec = pca.components_[1]

# Verify 90-degree perpendicularity via dot product and angle
dot_product = float(np.dot(pc1_vec, pc2_vec))
angle_deg = float(np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0))))

# Verify Pythagorean variance conservation
orig_total_var = float(np.var(X_centered, axis=0, ddof=1).sum())
pca_total_var = float(np.var(X_pca, axis=0, ddof=1).sum())

print(f"\nPC1 Eigenvector (Main Spread Axis):          {pc1_vec.round(4)}")
print(f"PC2 Eigenvector (Perpendicular 'Opposite'):   {pc2_vec.round(4)}")
print(f"Dot Product (PC1 · PC2):                     {dot_product:.6f} (Exact Zero = Orthogonal)")
print(f"Angle Between PC1 and PC2:                   {angle_deg:.2f}°")
print(f"\nOriginal Total Variance (Var_X + Var_Y):     {orig_total_var:.4f}")
print(f"Rotated Total Variance  (Var_PC1 + Var_PC2): {pca_total_var:.4f} (100% Conserved)")
print(f"Explained Variance Ratio -> PC1: {pca.explained_variance_ratio_[0]:.2%} | PC2: {pca.explained_variance_ratio_[1]:.2%}")

# -----------------------------------------------------------
# 3. PAIRWISE DISTANCE PRESERVATION (RED VS. GREEN POINTS)
# -----------------------------------------------------------
dist_2d = np.linalg.norm(X_centered[0] - X_centered[1])
dist_pc1 = abs(X_pca[0, 0] - X_pca[1, 0])
dist_pc2 = abs(X_pca[0, 1] - X_pca[1, 1])

print(f"\nDistance Between Reference Points (Red vs. Green):")
print(f"  Original 2D Euclidean Distance: {dist_2d:.3f}")
print(f"  Projected onto PC1 (1D):        {dist_pc1:.3f} (Preserves {dist_pc1 / dist_2d:.1%} of separation)")
print(f"  Projected onto PC2 (1D):        {dist_pc2:.3f} (Collapses to {dist_pc2 / dist_2d:.1%} — info lost!)")

# -----------------------------------------------------------
# 4. GENERATE 3-PANEL GEOMETRIC ROTATION VISUALIZATION
# -----------------------------------------------------------
os.makedirs("visualizations", exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

# Panel 1: Original Space with Orthogonal PC1 & PC2 Axes + Drop Projections
axes[0].scatter(X_centered[2:, 0], X_centered[2:, 1], alpha=0.55, color="steelblue", label="Flats")
axes[0].scatter(X_centered[0, 0], X_centered[0, 1], color="crimson", s=90, zorder=5, label="Point A (Red)")
axes[0].scatter(X_centered[1, 0], X_centered[1, 1], color="forestgreen", s=90, zorder=5, label="Point B (Green)")

# Draw PC1 and PC2 axis lines passing through origin (0, 0)
t_line = np.linspace(-2.6, 2.6, 100)
axes[0].plot(t_line * pc1_vec[0], t_line * pc1_vec[1], color="darkorange", lw=2.3, label="PC1 (Max Variance)")
axes[0].plot(t_line * pc2_vec[0], t_line * pc2_vec[1], color="purple", lw=2.0, linestyle="--", label="PC2 (90° Perpendicular)")

# Draw 90-degree orthogonal projection lines from sample points onto PC1
for idx in range(0, 25):
    pt = X_centered[idx]
    proj_pt = np.dot(pt, pc1_vec) * pc1_vec
    axes[0].plot([pt[0], proj_pt[0]], [pt[1], proj_pt[1]], color="gray", linestyle=":", alpha=0.55, lw=1)

axes[0].set_aspect("equal", adjustable="box")
axes[0].set_xlim(-2.6, 2.6)
axes[0].set_ylim(-2.6, 2.6)
axes[0].axhline(0, color="black", lw=0.7, alpha=0.4)
axes[0].axvline(0, color="black", lw=0.7, alpha=0.4)
axes[0].set_title("1. Original Space & 90° Rotated Axes")
axes[0].set_xlabel("Rooms (Centered)")
axes[0].set_ylabel("Washrooms (Centered)")
axes[0].legend(loc="upper left", fontsize=8)

# Panel 2: Rotated Coordinate System (PC1 vs. PC2)
axes[1].scatter(X_pca[2:, 0], X_pca[2:, 1], alpha=0.55, color="steelblue")
axes[1].scatter(X_pca[0, 0], X_pca[0, 1], color="crimson", s=90, zorder=5)
axes[1].scatter(X_pca[1, 0], X_pca[1, 1], color="forestgreen", s=90, zorder=5)
axes[1].axhline(0, color="darkorange", lw=2, label=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} Var)")
axes[1].axvline(0, color="purple", lw=1.8, linestyle="--", label=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} Var)")
axes[1].set_aspect("equal", adjustable="box")
axes[1].set_xlim(-2.8, 2.8)
axes[1].set_ylim(-2.8, 2.8)
axes[1].set_title("2. Rotated Frame (Axis Alignment)")
axes[1].set_xlabel("Principal Component 1 (PC1)")
axes[1].set_ylabel("Principal Component 2 (PC2)")
axes[1].legend(loc="upper left", fontsize=8.5)

# Panel 3: 1D Projection Comparison (Why High Variance = High Information)
axes[2].scatter(X_pca[2:, 0], np.ones(n_flats - 2) * 1.0, alpha=0.45, color="darkorange")
axes[2].scatter([X_pca[0, 0], X_pca[1, 0]], [1.0, 1.0], color=["crimson", "forestgreen"], s=95, zorder=5)

axes[2].scatter(X_pca[2:, 1], np.ones(n_flats - 2) * 0.0, alpha=0.45, color="purple")
axes[2].scatter([X_pca[0, 1], X_pca[1, 1]], [0.0, 0.0], color=["crimson", "forestgreen"], s=95, zorder=5)

axes[2].set_yticks([0.0, 1.0])
axes[2].set_yticklabels(["PC2 Only\n(Collapsed)", "PC1 Only\n(Preserved)"])
axes[2].set_ylim(-0.6, 1.6)
axes[2].set_xlabel("1D Projected Coordinate Value")
axes[2].set_title("3. 2D → 1D Projection Comparison")
axes[2].grid(axis="x", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("visualizations/pca_geometric_rotation.png")
plt.close()
print("\nSaved geometric visualization -> 'visualizations/pca_geometric_rotation.png'")