import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path whether run directly or via -m
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from modules.feature_engineering.transformers import ZScoreCapper, IQRCapper

print("=" * 68)
print("MASTER FEATURE ENGINEERING BENCHMARK (PRE-PCA CAPSTONE)")
print("=" * 68)

# -----------------------------------------------------------
# 1. GENERATE MESSY REAL-WORLD MIXED-TYPE DATASET
# -----------------------------------------------------------
np.random.seed(42)
n_samples = 1200

# Gaussian feature (Age) with MNAR missingness + extreme typos
age = np.random.normal(loc=32, scale=11, size=n_samples).clip(18, 70)
age[[10, 55, 190, 420]] = [145.0, 160.0, 2.0, 138.0]  # Outliers
age_missing_mask = (np.random.rand(n_samples) + (age / 180.0)) > 0.92
age_obs = age.copy()
age_obs[age_missing_mask] = np.nan

# Right-skewed feature (Fare) with extreme spikes
fare = np.random.gamma(shape=2.0, scale=22.0, size=n_samples) + 8.0
fare[[15, 88, 305, 710]] = [650.0, 820.0, 590.0, 740.0]

# Categorical feature with 45% structural missingness (CabinDeck)
cabin_deck = np.random.choice(["A", "B", "C", "D"], size=n_samples, p=[0.15, 0.25, 0.35, 0.25]).astype(object)
cabin_deck[np.random.rand(n_samples) < 0.45] = np.nan

# Discrete counts + untidy string field
sibsp = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.65, 0.25, 0.05, 0.03, 0.02])
parch = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.75, 0.15, 0.08, 0.02])

titles_true = np.random.choice(["Mr", "Mrs", "Miss", "Master"], size=n_samples, p=[0.55, 0.20, 0.18, 0.07])
names = [f"Passenger_{i}, {t}. Alex" for i, t in enumerate(titles_true)]

# Target variable driven by true clean signals + MNAR indicator + constructed features
family_size = sibsp + parch + 1
small_fam = ((family_size >= 2) & (family_size <= 4)).astype(int)
large_fam = (family_size >= 5).astype(int)
high_survival_title = np.isin(titles_true, ["Mrs", "Miss", "Master"]).astype(int)
has_cabin = (~pd.isna(cabin_deck)).astype(int)

logits = (
    -1.6
    + 0.015 * np.clip(fare, 8, 120)
    - 0.02 * np.clip(age, 18, 70)
    + 1.1 * age_missing_mask.astype(int)
    + 0.85 * has_cabin
    + 0.90 * small_fam
    - 1.30 * large_fam
    + 1.25 * high_survival_title
)
y = np.random.binomial(1, 1 / (1 + np.exp(-logits)), size=n_samples)

X = pd.DataFrame({
    "age": age_obs,
    "fare": fare,
    "cabin_deck": cabin_deck,
    "sibsp": sibsp,
    "parch": parch,
    "name": names,
})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 2. FEATURE CONSTRUCTION & SPLITTING HELPERS
# -----------------------------------------------------------
def build_family_type(df_in: pd.DataFrame) -> pd.DataFrame:
    size = df_in["sibsp"] + df_in["parch"] + 1
    fam_type = np.select([size == 1, (size >= 2) & (size <= 4)], [0, 1], default=2)
    return pd.DataFrame({"family_type": fam_type}, index=df_in.index)


def extract_title(df_in: pd.DataFrame) -> pd.DataFrame:
    titles = (
        df_in["name"]
        .str.split(",", expand=True)[1]
        .str.split(".", expand=True)[0]
        .str.strip()
    )
    return pd.DataFrame({"title": titles}, index=df_in.index)

# -----------------------------------------------------------
# 3. DEFINE THREE PROGRESSIVE PIPELINE STAGES
# -----------------------------------------------------------
# Stage 1: Naive Baseline (Median/Mode Imputation only, no outlier treatment or feature construction)
stage1_prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), ["age", "fare", "sibsp", "parch"]),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ]), ["cabin_deck"])
])

pipe_stage1 = Pipeline([
    ("prep", stage1_prep),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, random_state=42))
])

# Stage 2: Missing Indicators + Explicit 'Missing' Category + Hybrid Outlier Capping
stage2_prep = ColumnTransformer([
    ("age_pipe", Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("cap", ZScoreCapper(factor=3.0))
    ]), ["age"]),
    ("fare_pipe", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("cap", IQRCapper(factor=1.5))
    ]), ["fare"]),
    ("deck_pipe", Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ]), ["cabin_deck"]),
    ("counts", "passthrough", ["sibsp", "parch"])
])

pipe_stage2 = Pipeline([
    ("prep", stage2_prep),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, random_state=42))
])

# Stage 3: Full Feature Engineering (+ Domain Family Construction & Tidy Name Splitting)
stage3_prep = ColumnTransformer([
    ("age_pipe", Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("cap", ZScoreCapper(factor=3.0))
    ]), ["age"]),
    ("fare_pipe", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("cap", IQRCapper(factor=1.5))
    ]), ["fare"]),
    ("deck_pipe", Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ]), ["cabin_deck"]),
    ("family_ctor", Pipeline([
        ("build", FunctionTransformer(build_family_type)),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ]), ["sibsp", "parch"]),
    ("name_split", Pipeline([
        ("split", FunctionTransformer(extract_title)),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ]), ["name"])
])

pipe_stage3 = Pipeline([
    ("prep", stage3_prep),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, random_state=42))
])

# -----------------------------------------------------------
# 4. RUN 5-FOLD CROSS-VALIDATION & PLOT ABLATION RESULTS
# -----------------------------------------------------------
stages = {
    "Stage 1:\nNaive Baseline": pipe_stage1,
    "Stage 2:\nImpute + Outlier Cap": pipe_stage2,
    "Stage 3:\nFull FE + Construction": pipe_stage3,
}

results = []
for label, pipe in stages.items():
    scores = cross_validate(pipe, X_train, y_train, cv=5, scoring=["accuracy", "roc_auc"])
    pipe.fit(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)
    results.append({
        "Stage": label.replace("\n", " "),
        "CV_Acc": scores["test_accuracy"].mean() * 100,
        "CV_AUC": scores["test_roc_auc"].mean(),
        "Test_Acc": test_acc * 100
    })

results_df = pd.DataFrame(results)
print("\nAblation Benchmark Summary:")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

os.makedirs("visualizations", exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5))
x_pos = np.arange(len(stages))
bars = ax.bar(x_pos, results_df["CV_Acc"], color=["#94a3b8", "#3b82f6", "#10b981"], width=0.5, edgecolor="black")

ax.set_xticks(x_pos)
ax.set_xticklabels(list(stages.keys()))
ax.set_ylabel("5-Fold Cross-Validation Accuracy (%)")
ax.set_title("Progressive Impact of Feature Engineering Stages (Pre-PCA Capstone)")
ax.set_ylim(50, 95)
ax.grid(axis="y", linestyle=":", alpha=0.6)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("visualizations/master_pipeline_benchmark.png")
plt.close()
print("\nSaved benchmark chart -> 'visualizations/master_pipeline_benchmark.png'")