import os
import numpy as np
import matplotlib.pyplot as plt

print("=" * 68)
print("PCA PART 2 (STEP 1): COVARIANCE MATRIX & EIGENDECOMPOSITION PROOF")
print("=" * 68)

# -----------------------------------------------------------
# 1. GENERATE CORRELATED 2D DATASET & MEAN-CENTER
# -----------------------------------------------------------
np.random.seed(42)
n_samples = 250

# Create correlated 2D feature cloud
raw_x1 = np.random.normal(loc=50.0, scale=12.0, size=n_samples)
raw_x2 = 0.85 * raw_x1 + np.random.normal(loc=10.0, scale=4.5, size=n_samples)
X = np.column_stack([raw_x1, raw_x2])

# Mean-center the matrix: X_centered = X - mu
mean_vec = np.mean(X, axis=0)
X_centered = X - mean_vec

# -----------------------------------------------------------
# 2. MANUAL COVARIANCE MATRIX COMPUTATION
# -----------------------------------------------------------
# Formula: Sigma = (1 / (N - 1)) * (X_centered^T @ X_centered)
cov_manual = (X_centered.T @ X_centered) / (n_samples - 1)
cov_numpy = np.cov(X_centered, rowvar=False)

assert np.allclose(cov_manual, cov_numpy), "Manual covariance mismatch!"

print("\n--- 1. Symmetric Covariance Matrix (Sigma) ---")
print(np.round(cov_manual, 4))
print(f"Var(X1) = {cov_manual[0, 0]:.4f} | Var(X2) = {cov_manual[1, 1]:.4f} | Cov(X1, X2) = {cov_manual[0, 1]:.4f}")

# -----------------------------------------------------------
# 3. EIGENDECOMPOSITION & FUNDAMENTAL EQUATION VERIFICATION
# -----------------------------------------------------------
# Use eigh() for symmetric covariance matrices -> guarantees real eigenvalues & orthogonal vectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_manual)

# Sort in descending order of variance (largest eigenvalue first)
desc_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[desc_idx]
eigenvectors = eigenvectors[:, desc_idx]

v1, v2 = eigenvectors[:, 0], eigenvectors[:, 1]
lambda1, lambda2 = eigenvalues[0], eigenvalues[1]

# Verify Sigma @ v = lambda * v for both principal directions
lhs_1, rhs_1 = cov_manual @ v1, lambda1 * v1
lhs_2, rhs_2 = cov_manual @ v2, lambda2 * v2

assert np.allclose(lhs_1, rhs_1), "Eigenvector 1 equation failed!"
assert np.allclose(lhs_2, rhs_2), "Eigenvector 2 equation failed!"
assert np.isclose(np.dot(v1, v2), 0.0), "Eigenvectors are not orthogonal!"

print("\n--- 2. Eigendecomposition Verification (Sigma @ v = lambda * v) ---")
print(f"PC1 Eigenvalue (lambda_1): {lambda1:.4f} | Unit Eigenvector v1: {np.round(v1, 4)}")
print(f"  Check Sigma @ v1 = {np.round(lhs_1, 4)} == lambda_1 * v1 = {np.round(rhs_1, 4)}")
print(f"PC2 Eigenvalue (lambda_2): {lambda2:.4f} | Unit Eigenvector v2: {np.round(v2, 4)}")
print(f"  Check Sigma @ v2 = {np.round(lhs_2, 4)} == lambda_2 * v2 = {np.round(rhs_2, 4)}")
print(f"Explained Variance Ratio -> PC1: {lambda1 / (lambda1 + lambda2):.2%} | PC2: {lambda2 / (lambda1 + lambda2):.2%}")

# -----------------------------------------------------------
# 4. EMPIRICAL SWEEP: PROVING u^T @ Sigma @ u PEAKS AT v1
# -----------------------------------------------------------
angles_deg = np.linspace(0, 360, 720)
angles_rad = np.radians(angles_deg)
projected_variances = []

for theta in angles_rad:
    u = np.array([np.cos(theta), np.sin(theta)])
    # Project all samples onto unit vector u via dot product: X_centered @ u
    projections = X_centered @ u
    var_u = np.var(projections, ddof=1)
    projected_variances.append(var_u)

projected_variances = np.array(projected_variances)
max_empirical_var = float(projected_variances.max())
min_empirical_var = float(projected_variances.min())

print("\n--- 3. Empirical Unit-Vector Sweep (360° Search) ---")
print(f"Max Projected Variance across all angles: {max_empirical_var:.4f} (Matches lambda_1 = {lambda1:.4f})")
print(f"Min Projected Variance across all angles: {min_empirical_var:.4f} (Matches lambda_2 = {lambda2:.4f})")

# -----------------------------------------------------------
# 5. DIAGNOSTIC VISUALIZATION
# -----------------------------------------------------------
os.makedirs("visualizations", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

# Panel 1: Centered Data Cloud with Scaled Eigenvectors (v * sqrt(lambda))
axes[0].scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.45, color="steelblue", label="Centered Observations")
scale_1 = 2.0 * np.sqrt(lambda1)
scale_2 = 2.0 * np.sqrt(lambda2)

axes[0].quiver(0, 0, v1[0] * scale_1, v1[1] * scale_1, angles="xy", scale_units="xy", scale=1,
               color="crimson", width=0.008, label=f"v1 (λ1 = {lambda1:.1f})")
axes[0].quiver(0, 0, v2[0] * scale_2, v2[1] * scale_2, angles="xy", scale_units="xy", scale=1,
               color="darkorange", width=0.008, label=f"v2 (λ2 = {lambda2:.1f})")

axes[0].set_aspect("equal", adjustable="box")
axes[0].axhline(0, color="black", lw=0.7, alpha=0.4)
axes[0].axvline(0, color="black", lw=0.7, alpha=0.4)
axes[0].set_title("1. Eigenvectors of Covariance Matrix (Scaled by √λ)")
axes[0].set_xlabel("Feature 1 (Centered)")
axes[0].set_ylabel("Feature 2 (Centered)")
axes[0].legend(loc="upper left")
axes[0].grid(True, linestyle=":", alpha=0.5)

# Panel 2: Projected Variance as a Function of Unit Vector Angle
axes[1].plot(angles_deg, projected_variances, color="navy", lw=2.2, label="Var(X · u(θ))")
axes[1].axhline(lambda1, color="crimson", linestyle="--", lw=1.6, label=f"Max = λ1 ({lambda1:.2f})")
axes[1].axhline(lambda2, color="darkorange", linestyle="--", lw=1.6, label=f"Min = λ2 ({lambda2:.2f})")

axes[1].set_title("2. Objective Function: Projected Variance Across 360°")
axes[1].set_xlabel("Unit Vector Angle θ (Degrees)")
axes[1].set_ylabel("Projected Variance (uᵀ Σ u)")
axes[1].set_xlim(0, 360)
axes[1].set_xticks(np.arange(0, 361, 45))
axes[1].legend(loc="upper right")
axes[1].grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("visualizations/pca_eigen_geometry.png")
plt.close()
print("\nSaved diagnostic visualization -> 'visualizations/pca_eigen_geometry.png'")