import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

print("=" * 65)
print("FEATURE ENGINEERING: DATE AND TIME VARIABLES")
print("=" * 65)

# -----------------------------------------------------------
# 1. SYNTHETIC LOGISTICS & DELIVERY DATASET
# -----------------------------------------------------------
# Simulating delivery dispatches over several months:
# - order_timestamp: Raw string datetime
# - delivery_timestamp: Raw string datetime
# - distance_km: Continuous feature
# - delivery_duration_min: Target variable to predict

np.random.seed(42)
n_orders = 1000

# Generate synthetic timestamps across 2026
base_dates = pd.date_range(start="2026-01-01", end="2026-10-31", periods=n_orders)
random_offsets_hours = np.random.uniform(0, 24, size=n_orders)
order_times = base_dates + pd.to_timedelta(random_offsets_hours, unit="h")

# Distance between 2 and 45 km
distances = np.random.uniform(2, 45, size=n_orders)

# Delivery duration has rush hour delays and weekend variations
is_rush = ((order_times.hour >= 8) & (order_times.hour <= 10)) | ((order_times.hour >= 17) & (order_times.hour <= 20))
durations = (distances * 2.2) + (is_rush * 18.0) + np.random.normal(loc=10, scale=4, size=n_orders)
durations = np.maximum(durations, 12.0)  # Minimum 12 minutes

df = pd.DataFrame({
    "order_time": order_times.astype(str),
    "distance_km": distances,
    "delivery_duration_min": durations
})

print(f"Dataset generated: {df.shape[0]} orders, {df.shape[1]} columns.")
print("\nRaw Timestamp Sample:")
print(df.head(4))

# -----------------------------------------------------------
# 2. VECTORIZED DATETIME FEATURE EXTRACTION FUNCTION
# -----------------------------------------------------------
def extract_datetime_features(df_input):
    """
    Transforms raw timestamp strings into structured components,
    behavioral domain indicators, and cyclical trigonometric coordinates.
    """
    df_out = df_input.copy()
    
    # 1. Parse string timestamps into native pandas datetime
    order_dt = pd.to_datetime(df_out["order_time"])
    
    # 2. Component Extraction
    df_out["order_month"] = order_dt.dt.month
    df_out["order_day"] = order_dt.dt.day
    df_out["order_dayofweek"] = order_dt.dt.dayofweek
    df_out["order_hour"] = order_dt.dt.hour
    
    # 3. Behavioral Domain Flags
    df_out["is_weekend"] = (order_dt.dt.dayofweek >= 5).astype(int)
    df_out["is_month_end"] = order_dt.dt.is_month_end.astype(int)
    df_out["is_rush_hour"] = (
        ((order_dt.dt.hour >= 8) & (order_dt.dt.hour <= 10)) | 
        ((order_dt.dt.hour >= 17) & (order_dt.dt.hour <= 20))
    ).astype(int)
    
    # 4. Cyclical Trigonometric Encoding (Period: Hour=24, Month=12)
    df_out["hour_sin"] = np.sin(2 * np.pi * order_dt.dt.hour / 24.0)
    df_out["hour_cos"] = np.cos(2 * np.pi * order_dt.dt.hour / 24.0)
    
    df_out["month_sin"] = np.sin(2 * np.pi * order_dt.dt.month / 12.0)
    df_out["month_cos"] = np.cos(2 * np.pi * order_dt.dt.month / 12.0)
    
    # 5. Drop original raw datetime string column
    df_out.drop(columns=["order_time"], inplace=True)
    
    return df_out

# -----------------------------------------------------------
# 3. VERIFY CYCLICAL DISTANCES (HOUR 23 VS HOUR 0)
# -----------------------------------------------------------
print("\n--- 3. Mathematical Verification: Cyclical Proximity ---")

h23_sin, h23_cos = np.sin(2 * np.pi * 23 / 24.0), np.cos(2 * np.pi * 23 / 24.0)
h0_sin, h0_cos   = np.sin(2 * np.pi * 0 / 24.0),  np.cos(2 * np.pi * 0 / 24.0)
h12_sin, h12_cos = np.sin(2 * np.pi * 12 / 24.0), np.cos(2 * np.pi * 12 / 24.0)

dist_23_to_0 = np.sqrt((h23_sin - h0_sin)**2 + (h23_cos - h0_cos)**2)
dist_12_to_0 = np.sqrt((h12_sin - h0_sin)**2 + (h12_cos - h0_cos)**2)

print(f"Euclidean distance between Hour 23 and Hour 0:  {dist_23_to_0:.4f} (Very close!)")
print(f"Euclidean distance between Hour 12 and Hour 0:  {dist_12_to_0:.4f} (Opposite sides of circle)")

# -----------------------------------------------------------
# 4. PIPELINE ORCHESTRATION & LEAK-FREE SPLIT
# -----------------------------------------------------------
print("\n--- 4. End-to-End Pipeline Training ---")

X = df.drop(columns=["delivery_duration_min"])
y = df["delivery_duration_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Stateless scikit-learn transformer wrapping the extraction function
datetime_transformer = FunctionTransformer(extract_datetime_features)

full_pipeline = Pipeline([
    ("datetime_engineer", datetime_transformer),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit on raw training DataFrame containing string timestamps
full_pipeline.fit(X_train, y_train)

# Evaluate on holdout test partition
test_preds = full_pipeline.predict(X_test)
rmse = root_mean_squared_error(y_test, test_preds)
print(f"Pipeline Test RMSE: {rmse:.2f} minutes")

# -----------------------------------------------------------
# 5. VISUALIZING THE 24-HOUR CYCLICAL PROJECTION
# -----------------------------------------------------------
print("\n--- 5. Generating Cyclical Clock Visualizations ---")

hours = np.arange(24)
h_sin = np.sin(2 * np.pi * hours / 24.0)
h_cos = np.cos(2 * np.pi * hours / 24.0)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(h_cos, h_sin, color="navy", linestyle="--", alpha=0.6)
scatter = ax.scatter(h_cos, h_sin, c=hours, cmap="twilight", s=180, edgecolors="black", zorder=5)

for h, x_val, y_val in zip(hours, h_cos, h_sin):
    ax.text(x_val * 1.15, y_val * 1.15, f"{h}h", ha="center", va="center", fontsize=9, weight="bold")

ax.set_title("24-Hour Circular Projection (Sine vs. Cosine)", pad=20, fontsize=12)
ax.set_xlabel("Cos (Hour / 24)")
ax.set_ylabel("Sin (Hour / 24)")
ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
ax.set_aspect("equal")

plt.colorbar(scatter, ax=ax, label="Hour of Day (0 - 23)")
plt.tight_layout()
plt.savefig("visualizations/cyclical_time_encoding.png")
plt.close()
print("Saved cyclical visualization -> 'visualizations/cyclical_time_encoding.png'")

# -----------------------------------------------------------
# 6. INFERENCE TEST ON RAW PAYLOAD
# -----------------------------------------------------------
print("\n--- 6. Direct Inference on Unseen Raw Record ---")

raw_order = pd.DataFrame([{
    "order_time": "2026-11-01 18:30:00",  # Sunday peak rush-hour
    "distance_km": 14.5
}])

pred_duration = full_pipeline.predict(raw_order)[0]
print(f"Raw Input: {raw_order.to_dict(orient='records')[0]}")
print(f"Predicted Delivery Time: {pred_duration:.1f} minutes")