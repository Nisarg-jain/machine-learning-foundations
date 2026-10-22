import numpy as np
import pandas as pd

print("=" * 70)
print("DAY 19: UNDERSTANDING YOUR DATA (7 CORE QUESTIONS PIPELINE)")
print("=" * 70)

# Load Titanic dataset from raw GitHub mirror
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# -----------------------------------------------------------
# Question 1: How big is the data?
# -----------------------------------------------------------
print(f"\n1. DATA DIMENSIONS:")
print(f"   Total Rows:    {df.shape[0]}")
print(f"   Total Columns: {df.shape[1]}")

# -----------------------------------------------------------
# Question 2: How does the data look like?
# -----------------------------------------------------------
print("\n2. RANDOM SAMPLE (5 ROWS - AVOIDING ORDERING BIAS):")
print(df.sample(5, random_state=42)[["PassengerId", "Survived", "Pclass", "Name", "Age", "Fare"]])

# -----------------------------------------------------------
# Question 3: What are the column datatypes and memory footprint?
# -----------------------------------------------------------
print("\n3. COLUMN TYPES & MEMORY USAGE:")
print(df.info(memory_usage="deep"))

# -----------------------------------------------------------
# Question 4: Are there missing values?
# -----------------------------------------------------------
print("\n4. MISSING VALUE REPORT:")
missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    "Missing Count": missing_counts,
    "Percentage (%)": missing_pct.round(2)
})
print(missing_df[missing_df["Missing Count"] > 0])

# -----------------------------------------------------------
# Question 5: How does the data look mathematically?
# -----------------------------------------------------------
print("\n5. MATHEMATICAL 5-NUMBER SUMMARY (NUMERICAL FEATURES):")
print(df.describe().round(2))

# -----------------------------------------------------------
# Question 6: Are there duplicate records?
# -----------------------------------------------------------
duplicate_count = df.duplicated().sum()
print(f"\n6. DUPLICATE ROWS CHECK:")
print(f"   Total Duplicates Found: {duplicate_count}")

# -----------------------------------------------------------
# Question 7: How do features correlate with the target (Survived)?
# -----------------------------------------------------------
print("\n7. PEARSON CORRELATION WITH TARGET ('Survived'):")
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix["Survived"].sort_values(ascending=False).round(3))