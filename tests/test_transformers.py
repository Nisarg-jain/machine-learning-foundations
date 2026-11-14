import copy
import sys
from pathlib import Path

# Ensure repository root is on sys.path whether invoked via pytest or python directly
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modules.feature_engineering.transformers import (
    IQRCapper,
    RandomSampleImputer,
    Winsorizer,
    ZScoreCapper,
)


# -----------------------------------------------------------
# 1. ZERO DATA LEAKAGE & BOUND IMMUTABILITY
# -----------------------------------------------------------
def test_zero_data_leakage_on_transform():
    """
    Transforming unseen test data with massive outliers must clamp values
    using training bounds without mutating learned bounds_.
    """
    X_train = pd.DataFrame({
        "gaussian_col": np.linspace(50, 70, 100),
        "skewed_col": np.linspace(10, 40, 100),
    })
    X_test_extreme = pd.DataFrame({
        "gaussian_col": [-99999.0, 60.0, 99999.0],
        "skewed_col": [-50000.0, 25.0, 88888.0],
    })

    transformers = [
        ZScoreCapper(factor=3.0),
        IQRCapper(factor=1.5),
        Winsorizer(lower_quantile=0.05, upper_quantile=0.95),
    ]

    for transformer in transformers:
        transformer.fit(X_train)
        learned_bounds_snapshot = copy.deepcopy(transformer.bounds_)

        transformed = transformer.transform(X_test_extreme)

        # 1. Learned bounds must remain completely untouched after transform()
        assert transformer.bounds_ == learned_bounds_snapshot

        # 2. Extreme test points must be clamped strictly within training bounds
        for col_idx, col_name in enumerate(X_train.columns):
            lower = learned_bounds_snapshot[col_name]["lower"]
            upper = learned_bounds_snapshot[col_name]["upper"]
            assert transformed[:, col_idx].min() >= lower
            assert transformed[:, col_idx].max() <= upper


# -----------------------------------------------------------
# 2. RANDOM SAMPLE IMPUTER DETERMINISM & GUARDRAILS
# -----------------------------------------------------------
def test_random_sample_imputer_determinism_and_guardrails():
    """
    Verifies:
      - All NaNs are eliminated using only observed training values.
      - Repeated transform() calls with fixed random_state are deterministic.
      - Unfitted transform() raises NotFittedError.
      - All-NaN training column raises ValueError.
    """
    imputer = RandomSampleImputer(random_state=42)

    with pytest.raises(NotFittedError):
        imputer.transform(np.array([[1.0], [np.nan]]))

    with pytest.raises(ValueError):
        imputer.fit(pd.DataFrame({"empty_col": [np.nan, np.nan, np.nan]}))

    X_train = pd.DataFrame({"age": [22.0, 25.0, 30.0, np.nan, 45.0]})
    X_test = pd.DataFrame({"age": [np.nan, 28.0, np.nan, np.nan]})

    imputer.fit(X_train)
    out_first = imputer.transform(X_test)
    out_second = imputer.transform(X_test)

    # Zero NaNs remaining
    assert not np.isnan(out_first).any()

    # Deterministic output across repeated inference calls
    np.testing.assert_array_equal(out_first, out_second)

    # Imputed values come strictly from observed training pool
    valid_pool = {22.0, 25.0, 30.0, 45.0, 28.0}
    assert set(out_first.ravel()).issubset(valid_pool)


# -----------------------------------------------------------
# 3. EXACT MATHEMATICAL BOUNDARY VERIFICATION
# -----------------------------------------------------------
def test_exact_boundary_calculations():
    """
    Validates exact numerical fences for ZScoreCapper, IQRCapper, and Winsorizer.
    """
    # Known series: 0 to 100 inclusive (101 points -> Q1=25, Q3=75, IQR=50)
    series = pd.DataFrame({"val": np.arange(0, 101, dtype=float)})

    iqr_capper = IQRCapper(factor=1.5).fit(series)
    assert iqr_capper.bounds_["val"]["lower"] == pytest.approx(-50.0)
    assert iqr_capper.bounds_["val"]["upper"] == pytest.approx(150.0)

    winsorizer = Winsorizer(lower_quantile=0.10, upper_quantile=0.90).fit(series)
    assert winsorizer.bounds_["val"]["lower"] == pytest.approx(10.0)
    assert winsorizer.bounds_["val"]["upper"] == pytest.approx(90.0)

    z_capper = ZScoreCapper(factor=2.0).fit(series)
    expected_mean = float(series["val"].mean())
    expected_std = float(series["val"].std())
    assert z_capper.bounds_["val"]["lower"] == pytest.approx(expected_mean - 2.0 * expected_std)
    assert z_capper.bounds_["val"]["upper"] == pytest.approx(expected_mean + 2.0 * expected_std)


# -----------------------------------------------------------
# 4. SCIKIT-LEARN CLONE, PIPELINE & GRIDSEARCHCV COMPLIANCE
# -----------------------------------------------------------
def test_sklearn_pipeline_and_gridsearch_compatibility():
    """
    Confirms custom transformers integrate cleanly with sklearn.base.clone()
    and GridSearchCV hyperparameter tuning.
    """
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "f1": rng.normal(50, 10, size=80),
        "f2": rng.exponential(20, size=80),
    })
    y = 2.5 * X["f1"] - 1.2 * X["f2"] + rng.normal(0, 5, size=80)

    pipe = Pipeline([
        ("capper", IQRCapper(factor=1.5)),
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ])

    cloned_pipe = clone(pipe)
    assert cloned_pipe.get_params()["capper__factor"] == 1.5

    param_grid = {
        "capper": [IQRCapper(factor=1.5), IQRCapper(factor=2.0), Winsorizer(0.02, 0.98)],
        "model__alpha": [0.1, 1.0],
    }

    search = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="r2")
    search.fit(X, y)

    preds = search.predict(X.iloc[:5])
    assert preds.shape == (5,)
    assert not np.isnan(preds).any()


if __name__ == "__main__":
    # Direct fallback runner if pytest is not installed
    test_zero_data_leakage_on_transform()
    test_random_sample_imputer_determinism_and_guardrails()
    test_exact_boundary_calculations()
    test_sklearn_pipeline_and_gridsearch_compatibility()
    print("ALL 4 TRANSFORMER UNIT TESTS PASSED SUCCESSFULLY.")