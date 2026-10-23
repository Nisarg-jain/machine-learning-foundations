import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print("=" * 65)
print("BIVARIATE & MULTIVARIATE EXPLORATORY DATA ANALYSIS")
print("=" * 65)

# Load Seaborn benchmark datasets
tips = sns.load_dataset("tips")
titanic = sns.load_dataset("titanic")
iris = sns.load_dataset("iris")
flights = sns.load_dataset("flights")

# ===========================================================
# 1. NUMERICAL VS NUMERICAL (SCATTER & MULTI-FEATURE MAPPING)
# ===========================================================
print("\n--- 1. Numerical vs Numerical: Scatter & Line Analysis ---")

plt.figure(figsize=(10, 6))
# Encoding 5 dimensions into a single visualization:
# X: total_bill, Y: tip, Hue: sex, Style: smoker, Size: party size
sns.scatterplot(
    data=tips,
    x="total_bill",
    y="tip",
    hue="sex",
    style="smoker",
    size="size",
    sizes=(30, 200),
    alpha=0.85
)
plt.title("5D Multivariate Scatter: Total Bill vs Tip (Sex, Smoker, Table Size)")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")
plt.tight_layout()
plt.savefig("multivariate_scatter.png")
plt.close()
print("Saved 'multivariate_scatter.png'")

# Time-series line plot (Yearly flight passenger trend)
yearly_flights = flights.groupby("year")["passengers"].sum().reset_index()
plt.figure(figsize=(8, 4))
sns.lineplot(data=yearly_flights, x="year", y="passengers", marker="o", color="navy")
plt.title("Line Plot: Longitudinal Airline Passenger Trend (1949 - 1960)")
plt.xlabel("Year")
plt.ylabel("Total Passengers")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("bivariate_lineplot.png")
plt.close()
print("Saved 'bivariate_lineplot.png'")

# ===========================================================
# 2. NUMERICAL VS CATEGORICAL (BAR, BOX, & OVERLAPPING KDE)
# ===========================================================
print("\n--- 2. Numerical vs Categorical: Class Distributions ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# A. Bar plot: Average ticket fare per passenger class segmented by sex
sns.barplot(data=titanic, x="pclass", y="fare", hue="sex", ax=axes[0], palette="muted")
axes[0].set_title("Mean Fare by Class and Sex")
axes[0].set_xlabel("Passenger Class")
axes[0].set_ylabel("Average Fare")

# B. Box plot: Age distribution across classes segmented by survival
sns.boxplot(data=titanic, x="pclass", y="age", hue="survived", ax=axes[1], palette="Set2")
axes[1].set_title("Age Spread by Class and Survival")
axes[1].set_xlabel("Passenger Class")
axes[1].set_ylabel("Age")

# C. Overlapping KDE (PDF curves): Age distribution of Survived vs Died
sns.kdeplot(data=titanic[titanic["survived"] == 0]["age"], label="Died (0)", ax=axes[2], fill=True, color="red")
sns.kdeplot(data=titanic[titanic["survived"] == 1]["age"], label="Survived (1)", ax=axes[2], fill=True, color="green")
axes[2].set_title("Probability Density Function (PDF) of Age by Survival")
axes[2].set_xlabel("Age")
axes[2].legend()

plt.tight_layout()
plt.savefig("numerical_categorical_analysis.png")
plt.close()
print("Saved 'numerical_categorical_analysis.png'")

# ===========================================================
# 3. CATEGORICAL VS CATEGORICAL (CROSSTAB & HEATMAP)
# ===========================================================
print("\n--- 3. Categorical vs Categorical: Contingency & Heatmaps ---")

# Cross-tabulation of Survival rates across Passenger Classes
ct = pd.crosstab(titanic["pclass"], titanic["survived"])
print("Contingency Table (Pclass vs Survived):")
print(ct)

# Survival percentage breakdown per class
ct_pct = pd.crosstab(titanic["pclass"], titanic["survived"], normalize="index") * 100
print("\nPercentage Breakdown (%):")
print(ct_pct.round(2))

# Heatmap visualization
plt.figure(figsize=(7, 5))
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
plt.title("Heatmap: Cross-Tabulation of Class vs Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Passenger Class")
plt.tight_layout()
plt.savefig("categorical_heatmap.png")
plt.close()
print("Saved 'categorical_heatmap.png'")

# Pivot table and Hierarchical Clustermap on Flights (Month vs Year)
flight_matrix = flights.pivot(index="month", columns="year", values="passengers")
clustermap_fig = sns.clustermap(flight_matrix, cmap="viridis", figsize=(10, 8), standard_scale=1)
clustermap_fig.fig.suptitle("Hierarchical Clustermap: Seasonal Trends across Flight Months", y=1.02)
clustermap_fig.savefig("hierarchical_clustermap.png")
plt.close()
print("Saved 'hierarchical_clustermap.png'")

# ===========================================================
# 4. MULTI-FEATURE PAIR PLOT (IRIS DATASET)
# ===========================================================
print("\n--- 4. Multi-Feature Matrix: Pair Plot ---")
pair_grid = sns.pairplot(iris, hue="species", markers=["o", "s", "D"], palette="Dark2")
pair_grid.fig.subplots_adjust(top=0.95)
pair_grid.fig.suptitle("Pair Plot: All Numerical Pairwise Interactions (Iris)")
pair_grid.savefig("multivariate_pairplot.png")
plt.close()
print("Saved 'multivariate_pairplot.png'")

print("\nAll bivariate and multivariate visual pipelines executed successfully.")