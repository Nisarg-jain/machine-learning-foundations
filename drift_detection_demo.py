import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# -------------------------------------------------------------
# 1. Training Phase (Baseline Historical Distribution)
# -------------------------------------------------------------
print("=" * 60)
print("1. MODEL TRAINING & SERIALIZATION (Dev Phase)")
print("=" * 60)

np.random.seed(42)

# Normal baseline distribution: Mean CGPA = 7.5, Mean IQ = 105
X_cgpa = np.random.normal(loc=7.5, scale=0.8, size=100)
X_iq = np.random.normal(loc=105, scale=10, size=100)
X_train = np.column_stack((X_cgpa, X_iq))

# Target: 1 (Placed), 0 (Not Placed) based on combined threshold
y_train = (0.4 * X_cgpa + 0.6 * (X_iq / 15) > 7.0).astype(int)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save baseline statistics for drift tracking in production
baseline_stats = {
    "cgpa_mean": np.mean(X_cgpa),
    "cgpa_std": np.std(X_cgpa),
    "iq_mean": np.mean(X_iq),
    "iq_std": np.std(X_iq),
}

joblib.dump(model, "placement_model.joblib")
print("Model saved to 'placement_model.joblib'")
print(f"Baseline CGPA Mean: {baseline_stats['cgpa_mean']:.2f}")
print(f"Baseline IQ Mean:   {baseline_stats['iq_mean']:.2f}")

# -------------------------------------------------------------
# 2. Production Serving & Drift Detection Simulation
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("2. PRODUCTION INFERENCE & COVARIATE DRIFT MONITORING")
print("=" * 60)

# Simulate two production batches:
# Batch A: Normal incoming traffic (No Drift)
# Batch B: Distribution shift / Covariate Drift (e.g., student pool changes)
batch_normal = np.column_stack((
    np.random.normal(loc=7.45, scale=0.8, size=50),
    np.random.normal(loc=104.5, scale=10, size=50)
))

batch_drifted = np.column_stack((
    np.random.normal(loc=5.80, scale=0.8, size=50),  # Mean drops significantly
    np.random.normal(loc=88.0, scale=10, size=50)    # IQ drops significantly
))

def evaluate_batch(batch_data, batch_name, threshold=1.5):
    """
    Evaluates incoming production batch for covariate drift
    using standard deviation bounds from baseline.
    """
    batch_cgpa_mean = np.mean(batch_data[:, 0])
    batch_iq_mean = np.mean(batch_data[:, 1])

    cgpa_z_score = abs(batch_cgpa_mean - baseline_stats["cgpa_mean"]) / baseline_stats["cgpa_std"]
    iq_z_score = abs(batch_iq_mean - baseline_stats["iq_mean"]) / baseline_stats["iq_std"]

    print(f"\nEvaluating [{batch_name}]:")
    print(f"  Observed CGPA Mean: {batch_cgpa_mean:.2f} (Z-shift: {cgpa_z_score:.2f})")
    print(f"  Observed IQ Mean:   {batch_iq_mean:.2f} (Z-shift: {iq_z_score:.2f})")

    if cgpa_z_score > threshold or iq_z_score > threshold:
        print("  ALERT: Significant Data Drift Detected! Retraining pipeline triggered.")
    else:
        print("  Status: Data distribution stable. Serving live predictions safely.")

evaluate_batch(batch_normal, "Batch A (Live Traffic Day 1)")
evaluate_batch(batch_drifted, "Batch B (Live Traffic Day 15)")