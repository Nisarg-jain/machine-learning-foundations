import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import PCA as SklearnPCA

print("=" * 68)
print("PCA PART 2 (STEP 2): FROM-SCRATCH MANUAL PCA & 3D → 2D PROJECTION")
print("=" * 68)


class ManualPCA(BaseEstimator, TransformerMixin):
    """
    Production-grade Principal Component Analysis implemented from scratch
    via covariance matrix eigendecomposition.
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        n_samples, n_features = X_arr.shape

        if not (1 <= self.n_components <= n_features):
            raise ValueError(
                f"n_components={self.n_components} must be between 1 and n_features={n_features}."
            )

        # Step 1: Mean-center using training partition statistics only
        self.mean_ = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean_

        # Step 2: Compute symmetric d x d covariance matrix
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # Step 3: Eigendecomposition & descending variance sort
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        desc_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[desc_idx]
        eigenvectors = eigenvectors[:, desc_idx]

        # Deterministic sign alignment (matches Scikit-Learn SVD sign convention)
        max_abs_rows = np.argmax(np.abs(eigenvectors), axis=0)
        signs = np.sign(eigenvectors[max_abs_rows, range(n_features)])
        eigenvectors = eigenvectors * signs

        # Store top-k principal axes (shape: n_components x n_features)
        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        check_is_fitted(self, ["components_", "mean_"])
        X_arr = np.asarray(X, dtype=float)

        # Step 4: Project centered observations onto principal axes: (N x d) @ (d x k) -> (N x k)
        X_centered = X_arr - self.mean_
        return X_centered @ self.components_.T


# -----------------------------------------------------------
# 1. GENERATE SYNTHETIC 3D CLASSIFICATION DATASET (2 CLASSES)
# -----------------------------------------------------------
np.random.seed(42)
n_per_class = 60

# Class 0 cluster in 3D space
class_0 = np.random.multivariate_normal(
    mean=[2.0, 2.5, 1.5],
    cov=[[1.2, 0.8, 0.7], [0.8, 1.1, 0.6], [0.7, 0.6, 0.9]],
    size=n_per_class,
)

# Class 1 cluster shifted diagonally in 3D space
class_1 = np.random.multivariate_normal(
    mean=[5.5, 6.0, 4.8],
    cov=[[1.2, 0.8, 0.7], [0.8, 1.1, 0.6], [0.7, 0.6, 0.9]],
    size=n_per_class,
)

X_3d = np.vstack([class_0, class_1])
y_target = np.array([0] * n_per_class + [1] * n_per_class)

df_3d = pd.DataFrame(X_3d, columns=["feature_1", "feature_2", "feature_3"])
df_3d["target"] = y_target
print(f"\nOriginal 3D Dataset Shape: {df_3d.shape} (3 Features + 1 Target)")

# -----------------------------------------------------------
# 2. FIT MANUAL PCA & VERIFY PARITY AGAINST SCIKIT-LEARN
# -----------------------------------------------------------
manual_pca = ManualPCA(n_components=2)
Z_manual = manual_pca.fit_transform(X_3d)

sklearn_pca = SklearnPCA(n_components=2, random_state=42)
Z_sklearn = sklearn_pca.fit_transform(X_3d)

assert np.allclose(Z_manual, Z_sklearn, atol=1e-10), "Projected coordinates diverge from Sklearn!"
assert np.allclose(
    manual_pca.explained_variance_ratio_, sklearn_pca.explained_variance_ratio_, atol=1e-10
), "Explained variance ratio diverges from Sklearn!"

df_2d = pd.DataFrame(Z_manual, columns=["PC1", "PC2"])
df_2d["target"] = y_target

print(f"Transformed 2D Dataset Shape: {df_2d.shape} (PC1 + PC2 + Target)")
print(f"Manual PCA Explained Variance Ratio:  {np.round(manual_pca.explained_variance_ratio_, 5)}")
print(f"Sklearn PCA Explained Variance Ratio: {np.round(sklearn_pca.explained_variance_ratio_, 5)}")
print(f"Total Variance Retained in 2D:        {manual_pca.explained_variance_ratio_.sum():.2%}")
print("Parity Check: ManualPCA matches sklearn.decomposition.PCA to 1e-10 precision.")

# -----------------------------------------------------------
# 3. VISUALIZE 3D INPUT SPACE VS. 2D PCA SUBSPACE
# -----------------------------------------------------------
os.makedirs("visualizations", exist_ok=True)
fig = plt.figure(figsize=(14, 5.5))

# Left Subplot: Original 3D Scatter Plot
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
for cls, color, label in [(0, "royalblue", "Class 0"), (1, "crimson", "Class 1")]:
    mask = y_target == cls
    ax1.scatter(
        X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
        color=color, alpha=0.75, s=38, label=label
    )
ax1.set_title("1. Original 3D Feature Space (F1, F2, F3)")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.set_zlabel("Feature 3")
ax1.legend(loc="upper left")

# Right Subplot: Compressed 2D Principal Component Projection
ax2 = fig.add_subplot(1, 2, 2)
for cls, color, label in [(0, "royalblue", "Class 0"), (1, "crimson", "Class 1")]:
    mask = y_target == cls
    ax2.scatter(
        Z_manual[mask, 0], Z_manual[mask, 1],
        color=color, alpha=0.75, s=42, edgecolors="k", linewidths=0.4, label=label
    )
ax2.axhline(0, color="gray", linestyle=":", lw=1)
ax2.axvline(0, color="gray", linestyle=":", lw=1)
ax2.set_title(
    f"2. Compressed 2D PCA Subspace ({manual_pca.explained_variance_ratio_.sum():.1%} Variance Retained)"
)
ax2.set_xlabel(f"PC1 ({manual_pca.explained_variance_ratio_[0]:.1%} Variance)")
ax2.set_ylabel(f"PC2 ({manual_pca.explained_variance_ratio_[1]:.1%} Variance)")
ax2.legend(loc="upper left")
ax2.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("visualizations/pca_3d_to_2d_projection.png")
plt.close()
print("\nSaved 3D-to-2D projection figure -> 'visualizations/pca_3d_to_2d_projection.png'")