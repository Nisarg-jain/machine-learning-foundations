# Feature Engineering: Principles, Frameworks, and Core Pillars

---

## 1. Overview & Mathematical Necessity
* **Definition:** The process of applying domain knowledge and statistical transformations to raw data to expose clearer signals to machine learning algorithms.
* **Core Rule:** $\text{Simple Algorithm} + \text{Engineered Features} \gg \text{Complex Algorithm} + \text{Raw Features}$.
* Machine learning algorithms optimize mathematical cost functions over numeric feature spaces. They fail or degrade when fed unhandled missing values, raw string categoricals, multi-scale coordinate axes, or redundant uninformative dimensions.

---

## 2. The Four Pillars of Feature Engineering

### Pillar 1: Feature Transformation
Modifying existing individual columns into representations suitable for algorithmic consumption.
* **Missing Value Imputation:**
  - *Complete Case Analysis:* Dropping rows with null values (appropriate only when $<5\%$ of rows are missing at random).
  - *Univariate Imputation:* Replacing `NaN` with central tendencies (Mean/Median for numerical, Mode/Constant for categorical).
  - *Multivariate Imputation:* Predicting missing values using models (KNN Imputer, MICE / IterativeImputer).
* **Categorical Encoding:**
  - *One-Hot Encoding (OHE):* Converts nominal categories into $k$ or $k-1$ binary indicator columns.
  - *Ordinal Encoding:* Maps rank-ordered categories to numeric sequences ($1, 2, 3$).
* **Outlier Capping & Trimming:**
  - Mitigates outlier pull on gradient-based and distance-based estimators using IQR or Z-score boundaries.
* **Feature Scaling:**
  - Standardizes variable ranges to prevent large-magnitude features (e.g., Salary) from dominating Euclidean distance metrics over smaller-magnitude features (e.g., Age).

### Pillar 2: Feature Construction
Deriving entirely new composite features by combining raw columns using domain logic.
* Condenses fragmented data points into clear behavioural signals (e.g., combining `SibSp` and `Parch` into `family_size` and categorizing passengers as `Alone`, `Small_Family`, or `Large_Family`).

### Pillar 3: Feature Selection
Selecting the most informative subset of features while eliminating redundant, uninformative, or zero-variance inputs.
* Reduces training time, prevents the curse of dimensionality, and curbs overfitting.
* Methods include Variance Thresholding, Pearson/Spearman correlation pruning, and Recursive Feature Elimination (RFE).

### Pillar 4: Feature Extraction
Projecting a high-dimensional feature space into a lower-dimensional coordinate system.
* Unlike feature selection (which discards columns wholesale), extraction constructs a smaller set of composite dimensions that retain the bulk of the original data's variance.
* Core techniques include Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA).