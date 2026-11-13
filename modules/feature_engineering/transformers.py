import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class RandomSampleImputer(BaseEstimator, TransformerMixin):
    """
    Replaces missing values by drawing random samples with replacement
    from observed values in the training partition.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.n_features_in_ = X_df.shape[1]
        self.train_samples_ = {}
        for col in X_df.columns:
            observed = X_df[col].dropna().values
            if len(observed) == 0:
                raise ValueError(f"Column '{col}' contains only NaN values during fit().")
            self.train_samples_[col] = observed
        return self

    def transform(self, X):
        check_is_fitted(self, ["train_samples_"])
        X_df = pd.DataFrame(X).copy()
        rng = np.random.RandomState(self.random_state)

        for col in X_df.columns:
            null_mask = X_df[col].isna()
            null_count = int(null_mask.sum())
            if null_count > 0:
                replacements = rng.choice(
                    self.train_samples_[col], size=null_count, replace=True
                )
                X_df.loc[null_mask, col] = replacements
        return X_df.values


class ZScoreCapper(BaseEstimator, TransformerMixin):
    """
    Caps Gaussian features outside [mean - factor * std, mean + factor * std].
    Learns bounds strictly during fit() to prevent data leakage.
    """

    def __init__(self, factor: float = 3.0):
        self.factor = factor

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.n_features_in_ = X_df.shape[1]
        self.bounds_ = {}
        for col in X_df.columns:
            mean_val = float(X_df[col].mean())
            std_val = float(X_df[col].std())
            self.bounds_[col] = {
                "lower": mean_val - (self.factor * std_val),
                "upper": mean_val + (self.factor * std_val),
            }
        return self

    def transform(self, X):
        check_is_fitted(self, ["bounds_"])
        X_df = pd.DataFrame(X).copy()
        for col, bound in self.bounds_.items():
            X_df[col] = X_df[col].clip(lower=bound["lower"], upper=bound["upper"])
        return X_df.values


class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Caps skewed features outside Tukey's fences [Q1 - factor * IQR, Q3 + factor * IQR].
    Learns quartiles strictly during fit() to prevent data leakage.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.n_features_in_ = X_df.shape[1]
        self.bounds_ = {}
        for col in X_df.columns:
            q1 = float(X_df[col].quantile(0.25))
            q3 = float(X_df[col].quantile(0.75))
            iqr = q3 - q1
            self.bounds_[col] = {
                "lower": q1 - (self.factor * iqr),
                "upper": q3 + (self.factor * iqr),
            }
        return self

    def transform(self, X):
        check_is_fitted(self, ["bounds_"])
        X_df = pd.DataFrame(X).copy()
        for col, bound in self.bounds_.items():
            X_df[col] = X_df[col].clip(lower=bound["lower"], upper=bound["upper"])
        return X_df.values


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Caps continuous features at specified lower and upper empirical quantiles.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.n_features_in_ = X_df.shape[1]
        self.bounds_ = {}
        for col in X_df.columns:
            self.bounds_[col] = {
                "lower": float(X_df[col].quantile(self.lower_quantile)),
                "upper": float(X_df[col].quantile(self.upper_quantile)),
            }
        return self

    def transform(self, X):
        check_is_fitted(self, ["bounds_"])
        X_df = pd.DataFrame(X).copy()
        for col, bound in self.bounds_.items():
            X_df[col] = X_df[col].clip(lower=bound["lower"], upper=bound["upper"])
        return X_df.values