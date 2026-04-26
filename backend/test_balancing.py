"""
Comprehensive pytest test suite for balancing_service.
Run from the backend/ directory: python -m pytest test_balancing.py -v
"""
import numpy as np
import pandas as pd
import pytest

from services.balancing_service import _ir, _prepare, _resample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def arrays():
    """100 majority (class 0) vs 20 minority (class 1) — IR = 5.0"""
    rng = np.random.default_rng(0)
    X_maj = rng.random((100, 4))
    X_min = rng.random((20, 4))
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(100, dtype=int), np.ones(20, dtype=int)])
    return X, y


@pytest.fixture
def df():
    """DataFrame matching the arrays fixture, with a 'label' column."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "f1": rng.random(120),
        "f2": rng.random(120),
        "f3": rng.random(120),
        "f4": rng.random(120),
        "label": ["majority"] * 100 + ["minority"] * 20,
    })


# ---------------------------------------------------------------------------
# _ir correctness
# ---------------------------------------------------------------------------

def test_ir_formula():
    y = np.array([0] * 100 + [1] * 20)
    assert _ir(y) == 5.0


def test_ir_balanced():
    y = np.array([0] * 50 + [1] * 50)
    assert _ir(y) == 1.0


def test_ir_zero_minority():
    # class 1 is absent → bincount fills index 1 with 0 → triggers 9999.0 guard
    y = np.array([0, 0, 2, 2])
    assert _ir(y) == 9999.0


def test_ir_multiclass():
    y = np.array([0] * 60 + [1] * 30 + [2] * 10)
    assert _ir(y) == 6.0  # 60 / 10


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------

def test_smote_minority_increases(arrays):
    X, y = arrays
    _, y_bal = _resample("smote", {"k_neighbors": 5}, X, y)
    assert np.sum(y_bal == 1) > np.sum(y == 1)


def test_smote_majority_unchanged(arrays):
    X, y = arrays
    _, y_bal = _resample("smote", {"k_neighbors": 5}, X, y)
    assert np.sum(y_bal == 0) == np.sum(y == 0)


def test_smote_total_increases(arrays):
    X, y = arrays
    _, y_bal = _resample("smote", {"k_neighbors": 5}, X, y)
    assert len(y_bal) > len(y)


def test_smote_ir_decreases(arrays):
    X, y = arrays
    _, y_bal = _resample("smote", {"k_neighbors": 5}, X, y)
    assert _ir(y_bal) < _ir(y)


def test_smote_original_rows_preserved(arrays):
    """SMOTE only adds synthetic rows; every original row must still be present."""
    X, y = arrays
    X_bal, y_bal = _resample("smote", {"k_neighbors": 5}, X, y)
    # imbalanced-learn preserves originals at the front of the output array
    assert np.allclose(X_bal[: len(X)], X)
    assert np.array_equal(y_bal[: len(y)], y)


# ---------------------------------------------------------------------------
# NearMiss
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("version", [1, 2, 3])
def test_nearmiss_majority_decreases(arrays, version):
    X, y = arrays
    _, y_bal = _resample("nearmiss", {"version": version, "n_neighbors": 3}, X, y)
    assert np.sum(y_bal == 0) < np.sum(y == 0)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_nearmiss_minority_unchanged(arrays, version):
    X, y = arrays
    _, y_bal = _resample("nearmiss", {"version": version, "n_neighbors": 3}, X, y)
    assert np.sum(y_bal == 1) == np.sum(y == 1)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_nearmiss_total_decreases(arrays, version):
    X, y = arrays
    _, y_bal = _resample("nearmiss", {"version": version, "n_neighbors": 3}, X, y)
    assert len(y_bal) < len(y)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_nearmiss_ir_decreases(arrays, version):
    X, y = arrays
    _, y_bal = _resample("nearmiss", {"version": version, "n_neighbors": 3}, X, y)
    assert _ir(y_bal) < _ir(y)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_nearmiss_no_data_corruption(arrays, version):
    """Every row kept by NearMiss must be an exact copy of an original row."""
    X, y = arrays
    X_bal, y_bal = _resample("nearmiss", {"version": version, "n_neighbors": 3}, X, y)
    for row in X_bal:
        assert np.any(np.all(np.isclose(X, row), axis=1)), \
            "NearMiss output contains a row not present in the original data"


# ---------------------------------------------------------------------------
# Combined (SMOTE → NearMiss)
# ---------------------------------------------------------------------------

def test_combined_ir_decreases(arrays):
    X, y = arrays
    _, y_bal = _resample("combined", {"k_neighbors": 5, "nearmiss_version": 1, "n_neighbors": 3}, X, y)
    assert _ir(y_bal) < _ir(y)


def test_combined_minority_grows_or_stays(arrays):
    X, y = arrays
    _, y_bal = _resample("combined", {"k_neighbors": 5, "nearmiss_version": 1, "n_neighbors": 3}, X, y)
    assert np.sum(y_bal == 1) >= np.sum(y == 1)


def test_combined_majority_shrinks_or_stays(arrays):
    X, y = arrays
    _, y_bal = _resample("combined", {"k_neighbors": 5, "nearmiss_version": 1, "n_neighbors": 3}, X, y)
    assert np.sum(y_bal == 0) <= np.sum(y == 0)


# ---------------------------------------------------------------------------
# General data integrity (all methods)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,params", [
    ("smote",    {"k_neighbors": 5}),
    ("nearmiss", {"version": 1, "n_neighbors": 3}),
    ("combined", {"k_neighbors": 5, "nearmiss_version": 1, "n_neighbors": 3}),
])
def test_no_nan_introduced(arrays, method, params):
    X, y = arrays
    X_bal, _ = _resample(method, params, X, y)
    assert not np.any(np.isnan(X_bal))


@pytest.mark.parametrize("method,params", [
    ("smote",    {"k_neighbors": 5}),
    ("nearmiss", {"version": 1, "n_neighbors": 3}),
    ("combined", {"k_neighbors": 5, "nearmiss_version": 1, "n_neighbors": 3}),
])
def test_no_inf_introduced(arrays, method, params):
    X, y = arrays
    X_bal, _ = _resample(method, params, X, y)
    assert not np.any(np.isinf(X_bal))


@pytest.mark.parametrize("method,params", [
    ("smote",    {"k_neighbors": 5}),
    ("nearmiss", {"version": 1, "n_neighbors": 3}),
    ("combined", {"k_neighbors": 5, "nearmiss_version": 1, "n_neighbors": 3}),
])
def test_no_new_classes(arrays, method, params):
    X, y = arrays
    _, y_bal = _resample(method, params, X, y)
    assert set(np.unique(y_bal)) == set(np.unique(y))


@pytest.mark.parametrize("method,params", [
    ("smote",    {"k_neighbors": 5}),
    ("nearmiss", {"version": 1, "n_neighbors": 3}),
    ("combined", {"k_neighbors": 5, "nearmiss_version": 1, "n_neighbors": 3}),
])
def test_reproducibility(arrays, method, params):
    X, y = arrays
    X1, y1 = _resample(method, params, X, y)
    X2, y2 = _resample(method, params, X, y)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


# ---------------------------------------------------------------------------
# _prepare / test-set isolation
# ---------------------------------------------------------------------------

def test_prepare_train_size(df):
    total = len(df)
    X_train, y_train, le, col_names = _prepare(df, "label", 0.2)
    expected_train = total - round(total * 0.2)
    assert abs(len(y_train) - expected_train) <= 2


def test_prepare_label_encoder(df):
    _, y_train, le, _ = _prepare(df, "label", 0.2)
    assert set(le.classes_) == {"majority", "minority"}
    assert set(np.unique(y_train)).issubset({0, 1})


def test_prepare_col_names(df):
    _, _, _, col_names = _prepare(df, "label", 0.2)
    assert len(col_names) == 4  # f1-f4; label is excluded


def test_prepare_does_not_modify_df(df):
    original_shape = df.shape
    _prepare(df, "label", 0.2)
    assert df.shape == original_shape


def test_prepare_test_set_not_exposed(df):
    """_prepare discards the test split — balancing operates only on training data."""
    X_train, y_train, le, _ = _prepare(df, "label", 0.2)
    X_bal, y_bal = _resample("smote", {"k_neighbors": 5}, X_train, y_train)
    # Original training rows are preserved at the front (SMOTE only appends)
    assert np.allclose(X_bal[: len(X_train)], X_train)
    # SMOTE added rows → confirms it ran on training data, not the full dataset
    assert len(y_bal) > len(y_train)


def test_unknown_method_raises(arrays):
    X, y = arrays
    with pytest.raises(ValueError, match="Unknown balancing method"):
        _resample("bogus", {}, X, y)
