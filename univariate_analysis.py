import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

print("=" * 65)
print("EXPLORATORY DATA ANALYSIS: UNIVARIATE PROFILING")
print("=" * 65)

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ===========================================================
# 1. CATEGORICAL ANALYSIS: COUNTS & PERCENTAGES
# ===========================================================
print("\n--- 1. CATEGORICAL FEATURE FREQUENCY TABLES ---")

categorical_cols = ["Survived", "Pclass", "Sex", "Embarked"]

for col in categorical_cols:
  counts = df[col].value_counts(dropna=False)
  proportions = df[col].value_counts(normalize=True, dropna=False) * 100
  summary_table = pd.DataFrame(
      {"Count": counts, "Percentage (%)": proportions.round(2)}
  )
  print(f"\nFeature: {col}")
  print(summary_table)

# Plotting Categorical Features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Survived countplot
sns.countplot(x="Survived", data=df, ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("Survived Distribution (0 = Died, 1 = Survived)")

# Pclass countplot
sns.countplot(x="Pclass", data=df, ax=axes[0, 1], palette="Set1")
axes[0, 1].set_title("Passenger Class Distribution")

# Sex countplot
sns.countplot(x="Sex", data=df, ax=axes[1, 0], palette="pastel")
axes[1, 0].set_title("Gender Breakdown")

# Embarked pie chart
df["Embarked"].value_counts().plot(
    kind="pie",
    autopct="%.2f%%",
    ax=axes[1, 1],
    startangle=90,
    colors=["#66b3ff", "#99ff99", "#ffcc99"],
)
axes[1, 1].set_title("Port of Embarkation Proportions")
axes[1, 1].set_ylabel("")

plt.tight_layout()
plt.savefig("categorical_univariate_plots.png")
plt.close()
print(
    "\nSaved categorical visualization grid as 'categorical_univariate_plots.png'"
)

# ===========================================================
# 2. NUMERICAL ANALYSIS: SPREAD, SHAPE, & OUTLIERS
# ===========================================================
print("\n--- 2. NUMERICAL FEATURE STATISTICAL METRICS ---")

numerical_cols = ["Age", "Fare"]

for col in numerical_cols:
  q1 = df[col].quantile(0.25)
  median = df[col].median()
  q3 = df[col].quantile(0.75)
  iqr = q3 - q1
  lower_whisker = q1 - 1.5 * iqr
  upper_whisker = q3 + 1.5 * iqr
  skew = df[col].skew()

  print(f"\nFeature: {col}")
  print(
      f"  Mean: {df[col].mean():.2f} | Median: {median:.2f} | Std Dev:"
      f" {df[col].std():.2f}"
  )
  print(f"  Q1 (25%): {q1:.2f} | Q3 (75%): {q3:.2f} | IQR: {iqr:.2f}")
  print(f"  Outlier Thresholds: [< {lower_whisker:.2f} or > {upper_whisker:.2f}]")
  print(f"  Skewness: {skew:.2f}")

# Plotting Numerical Features (Histogram + KDE and Boxplots)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age Histogram + KDE
sns.histplot(
    df["Age"].dropna(), kde=True, ax=axes[0, 0], color="royalblue", bins=20
)
axes[0, 0].set_title("Age Distribution (Histogram + KDE Curve)")

# Age Boxplot
sns.boxplot(x=df["Age"], ax=axes[0, 1], color="lightblue")
axes[0, 1].set_title("Age Boxplot (Outlier Inspection)")

# Fare Histogram + KDE
sns.histplot(df["Fare"], kde=True, ax=axes[1, 0], color="crimson", bins=30)
axes[1, 0].set_title("Fare Distribution (Positive Skew)")

# Fare Boxplot
sns.boxplot(x=df["Fare"], ax=axes[1, 1], color="salmon")
axes[1, 1].set_title("Fare Boxplot (Extreme Fare Outliers)")

plt.tight_layout()
plt.savefig("numerical_univariate_plots.png")
plt.close()
print("Saved numerical visualization grid as 'numerical_univariate_plots.png'")