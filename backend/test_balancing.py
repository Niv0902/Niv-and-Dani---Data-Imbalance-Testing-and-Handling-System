"""
Comprehensive pytest test suite for balancing_service.
Run from the backend/ directory: python -m pytest test_balancing.py -v
"""
import numpy as np
import pandas as pd
import pytest

from services.balancing_service import _ir, _prepare, _resample


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def find_smote_parents(syn, X_orig_min, tol=1e-6):
    """
    Exhaustive search over all (A, B) pairs to find the two original minority
    rows that SMOTE interpolated to create `syn`.
    Returns (a_idx, b_idx, t) or (None, None, None) if not found.
    """
    n = len(X_orig_min)
    for a_idx in range(n):
        A = X_orig_min[a_idx]
        for b_idx in range(n):
            if a_idx == b_idx:
                continue
            B = X_orig_min[b_idx]
            diff = B - A
            nonzero = np.abs(diff) > 1e-10
            if not np.any(nonzero):
                continue
            t_vals = (syn[nonzero] - A[nonzero]) / diff[nonzero]
            if np.max(t_vals) - np.min(t_vals) > tol:
                continue          # t inconsistent across dimensions → wrong pair
            t = float(np.mean(t_vals))
            if not (0 < t < 1):
                continue
            if np.linalg.norm(A + t * (B - A) - syn) < tol:
                return a_idx, b_idx, t
    return None, None, None


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


@pytest.fixture
def tiny_minority():
    """10 majority, 3 minority — used to test the k-guard (k=5 > minority-1=2)."""
    rng = np.random.default_rng(1)
    X_maj = rng.random((10, 4))
    X_min = rng.random((3, 4))
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(10, dtype=int), np.ones(3, dtype=int)])
    return X, y


@pytest.fixture
def multiclass_arrays():
    """60 class-0, 30 class-1, 10 class-2 — IR = 6.0"""
    rng = np.random.default_rng(2)
    X = np.vstack([rng.random((60, 4)), rng.random((30, 4)), rng.random((10, 4))])
    y = np.concatenate([np.zeros(60, dtype=int), np.ones(30, dtype=int), np.full(10, 2, dtype=int)])
    return X, y


@pytest.fixture
def cat_df():
    """DataFrame with one numeric and one categorical column — for encode/decode tests."""
    rng = np.random.default_rng(3)
    n_maj, n_min = 60, 12
    return pd.DataFrame({
        "age":       np.concatenate([rng.integers(20, 65, n_maj), rng.integers(20, 65, n_min)]).astype(float),
        "workclass": (["Private"] * 30 + ["Self-emp"] * 20 + ["Govt"] * 10) + (["Private"] * 6 + ["Govt"] * 6),
        "label":     ["majority"] * n_maj + ["minority"] * n_min,
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
    # class 1 absent → bincount gap → 9999.0 guard
    y = np.array([0, 0, 2, 2])
    assert _ir(y) == 9999.0


def test_ir_multiclass():
    y = np.array([0] * 60 + [1] * 30 + [2] * 10)
    assert _ir(y) == 6.0


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------

def test_smote_minority_increases(arrays):
    X, y = arrays
    _, y_bal, _ = _resample("smote", {"k_neighbors": 5}, X, y)
    assert np.sum(y_bal == 1) > np.sum(y == 1)


def test_smote_minority_matches_majority(arrays):
    """SMOTE oversamples minority until it equals the majority count."""
    X, y = arrays
    _, y_bal, _ = _resample("smote", {"k_neighbors": 5}, X, y)
    assert np.sum(y_bal == 1) == np.sum(y_bal == 0)


def test_smote_majority_count_unchanged(arrays):
    X, y = arrays
    _, y_bal, _ = _resample("smote", {"k_neighbors": 5}, X, y)
    assert np.sum(y_bal == 0) == np.sum(y == 0)


def test_smote_majority_rows_in_same_order(arrays):
    """Every original majority row is still present and in the same order."""
    X, y = arrays
    X_bal, y_bal, _ = _resample("smote", {"k_neighbors": 5}, X, y)
    X_orig_maj = X[y == 0]
    # imbalanced-learn places originals first; the first n_majority rows must be the majority
    assert np.allclose(X_bal[:len(X_orig_maj)], X_orig_maj)
    assert np.all(y_bal[:len(X_orig_maj)] == 0)


def test_smote_total_increases(arrays):
    X, y = arrays
    _, y_bal, _ = _resample("smote", {"k_neighbors": 5}, X, y)
    assert len(y_bal) > len(y)


def test_smote_ir_decreases(arrays):
    X, y = arrays
    _, y_bal, _ = _resample("smote", {"k_neighbors": 5}, X, y)
    assert _ir(y_bal) < _ir(y)


def test_smote_original_rows_preserved(arrays):
    """All original rows (both classes) appear at the front of X_bal, unchanged."""
    X, y = arrays
    X_bal, y_bal, _ = _resample("smote", {"k_neighbors": 5}, X, y)
    assert np.allclose(X_bal[:len(X)], X)
    assert np.array_equal(y_bal[:len(y)], y)


def test_smote_no_exact_duplicates(arrays):
    """Synthetic rows must not be exact copies of any original minority row."""
    X, y = arrays
    X_bal, y_bal, _ = _resample("smote", {"k_neighbors": 5}, X, y)
    X_orig_min = X[y == 1]
    synthetic_X = X_bal[len(X):]           # rows added by SMOTE
    for row in synthetic_X:
        assert not np.any(np.all(np.isclose(X_orig_min, row), axis=1)), \
            f"Synthetic row {row} is an exact duplicate of an original minority row"


def test_smote_synthetic_are_interpolations(arrays):
    """
    Each synthetic row must lie exactly on the line between two original minority rows:
      synthetic = A + t * (B - A)   for some 0 < t < 1
    Verified using exhaustive parent search across all (A, B) pairs.
    """
    X, y = arrays
    X_bal, y_bal, _ = _resample("smote", {"k_neighbors": 5}, X, y)
    X_orig_min = X[y == 1]
    synthetic_X = X_bal[len(X):]

    for i, syn in enumerate(synthetic_X):
        a_idx, b_idx, t = find_smote_parents(syn, X_orig_min)
        assert a_idx is not None, (
            f"Synthetic row {i} [{syn}] could not be reconstructed from any pair "
            f"of original minority rows — it is NOT a genuine SMOTE interpolation."
        )
        assert 0 < t < 1, f"Interpolation weight t={t} is out of range (0, 1)"


def test_smote_k_guard_small_minority(tiny_minority):
    """
    k_neighbors=5 with only 3 minority samples: guard must clamp k to 2
    (minority_count - 1 = 2) and not raise an exception.
    """
    X, y = tiny_minority
    # minority has 3 samples; k=5 would crash without the guard
    X_bal, y_bal, _ = _resample("smote", {"k_neighbors": 5}, X, y)
    assert np.sum(y_bal == 1) >= np.sum(y == 1)   # minority grew or stayed


def test_smote_reproducibility(arrays):
    X, y = arrays
    X1, y1, _ = _resample("smote", {"k_neighbors": 5}, X, y)
    X2, y2, _ = _resample("smote", {"k_neighbors": 5}, X, y)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_smote_categorical_roundtrip(cat_df):
    """
    After SMOTE, the decoded categorical column must contain only valid original
    category labels — never a raw integer or an out-of-vocabulary value.
    """
    X_train, y_train, le, col_names, enc, cat_cols, numeric_cols, *_ = _prepare(cat_df, "label")
    assert enc is not None, "Expected an OrdinalEncoder for the categorical column"

    X_bal, y_bal, _ = _resample("smote", {"k_neighbors": 3}, X_train, y_train)

    # Replicate the decode logic from _pipeline
    n_num = len(numeric_cols)
    cat_part = X_bal[:, n_num:]
    cat_int = np.empty(cat_part.shape, dtype=int)
    for i, cats in enumerate(enc.categories_):
        cat_int[:, i] = np.clip(np.round(cat_part[:, i]).astype(int), 0, len(cats) - 1)
    decoded = enc.inverse_transform(cat_int)

    valid_categories = set(enc.categories_[0])   # "workclass" column
    for val in decoded[:, 0]:
        assert val in valid_categories, (
            f"Decoded value {val!r} is not a valid category. "
            f"Valid: {valid_categories}"
        )


# ---------------------------------------------------------------------------
# NearMiss (version 1 — hardcoded in backend)
# ---------------------------------------------------------------------------

def test_nearmiss_minority_unchanged(arrays):
    X, y = arrays
    _, y_bal, _ = _resample("nearmiss", {"n_neighbors": 3}, X, y)
    assert np.sum(y_bal == 1) == np.sum(y == 1)


def test_nearmiss_majority_matches_minority(arrays):
    """NearMiss reduces majority until majority_count == minority_count."""
    X, y = arrays
    _, y_bal, _ = _resample("nearmiss", {"n_neighbors": 3}, X, y)
    assert np.sum(y_bal == 0) == np.sum(y_bal == 1)


def test_nearmiss_majority_decreases(arrays):
    X, y = arrays
    _, y_bal, _ = _resample("nearmiss", {"n_neighbors": 3}, X, y)
    assert np.sum(y_bal == 0) < np.sum(y == 0)


def test_nearmiss_total_decreases(arrays):
    X, y = arrays
    _, y_bal, _ = _resample("nearmiss", {"n_neighbors": 3}, X, y)
    assert len(y_bal) < len(y)


def test_nearmiss_ir_decreases(arrays):
    X, y = arrays
    _, y_bal, _ = _resample("nearmiss", {"n_neighbors": 3}, X, y)
    assert _ir(y_bal) < _ir(y)


def test_nearmiss_no_data_corruption(arrays):
    """Every row kept by NearMiss must be an exact copy of an original row."""
    X, y = arrays
    X_bal, _, _ = _resample("nearmiss", {"n_neighbors": 3}, X, y)
    for row in X_bal:
        assert np.any(np.all(np.isclose(X, row), axis=1)), \
            "NearMiss output contains a row not found in the original data"


def test_nearmiss_determinism(arrays):
    """Same input must always produce the same output."""
    X, y = arrays
    X1, y1, _ = _resample("nearmiss", {"n_neighbors": 3}, X, y)
    X2, y2, _ = _resample("nearmiss", {"n_neighbors": 3}, X, y)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_nearmiss_multiclass(multiclass_arrays):
    """NearMiss must work correctly on a 3-class dataset and reduce IR."""
    X, y = multiclass_arrays
    X_bal, y_bal, _ = _resample("nearmiss", {"n_neighbors": 3}, X, y)
    assert _ir(y_bal) < _ir(y)
    assert set(np.unique(y_bal)) == {0, 1, 2}   # all classes still present


# ---------------------------------------------------------------------------
# Combined (SMOTE → NearMiss)
# ---------------------------------------------------------------------------

def test_combined_ir_decreases(arrays):
    X, y = arrays
    _, y_bal, _ = _resample("combined", {"k_neighbors": 5, "n_neighbors": 3}, X, y)
    assert _ir(y_bal) < _ir(y)


def test_combined_minority_grows(arrays):
    """SMOTE ran → minority count is strictly larger than original."""
    X, y = arrays
    _, y_bal, _ = _resample("combined", {"k_neighbors": 5, "n_neighbors": 3}, X, y)
    assert np.sum(y_bal == 1) > np.sum(y == 1)


def test_combined_majority_does_not_grow(arrays):
    """NearMiss only removes majority rows — the final majority count must be ≤ original."""
    X, y = arrays
    _, y_bal, _ = _resample("combined", {"k_neighbors": 5, "n_neighbors": 3}, X, y)
    assert np.sum(y_bal == 0) <= np.sum(y == 0)


def test_combined_both_methods_contribute(arrays):
    """
    Combined uses a midpoint strategy so both methods do real work:
    SMOTE grows minority from original count toward majority,
    NearMiss shrinks majority down to meet it.
    Verifies minority grew (SMOTE ran) AND majority shrank (NearMiss ran).
    """
    X, y = arrays
    _, y_comb, _ = _resample("combined", {"k_neighbors": 5, "n_neighbors": 3}, X, y)
    assert np.sum(y_comb == 1) > np.sum(y == 1), \
        "SMOTE should have grown the minority class"
    assert np.sum(y_comb == 0) < np.sum(y == 0), \
        "NearMiss should have reduced the majority class"


def test_combined_ir_close_to_one(arrays):
    """After SMOTE + NearMiss the dataset should be near-balanced (IR ≤ 1.5)."""
    X, y = arrays
    _, y_bal, _ = _resample("combined", {"k_neighbors": 5, "n_neighbors": 3}, X, y)
    assert _ir(y_bal) <= 1.5, f"IR after Combined = {_ir(y_bal):.2f}, expected ≤ 1.5"


def test_combined_reproducibility(arrays):
    X, y = arrays
    params = {"k_neighbors": 5, "n_neighbors": 3}
    X1, y1, _ = _resample("combined", params, X, y)
    X2, y2, _ = _resample("combined", params, X, y)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


# ---------------------------------------------------------------------------
# General integrity — all methods
# ---------------------------------------------------------------------------

ALL_METHOD_PARAMS = [
    ("smote",    {"k_neighbors": 5}),
    ("nearmiss", {"n_neighbors": 3}),
    ("combined", {"k_neighbors": 5, "n_neighbors": 3}),
]


@pytest.mark.parametrize("method,params", ALL_METHOD_PARAMS)
def test_no_nan_introduced(arrays, method, params):
    X, y = arrays
    X_bal, _, _ = _resample(method, params, X, y)
    assert not np.any(np.isnan(X_bal))


@pytest.mark.parametrize("method,params", ALL_METHOD_PARAMS)
def test_no_inf_introduced(arrays, method, params):
    X, y = arrays
    X_bal, _, _ = _resample(method, params, X, y)
    assert not np.any(np.isinf(X_bal))


@pytest.mark.parametrize("method,params", ALL_METHOD_PARAMS)
def test_no_new_classes(arrays, method, params):
    X, y = arrays
    _, y_bal, _ = _resample(method, params, X, y)
    assert set(np.unique(y_bal)) == set(np.unique(y))


@pytest.mark.parametrize("method,params", ALL_METHOD_PARAMS)
def test_ir_after_lower_than_before(arrays, method, params):
    X, y = arrays
    _, y_bal, _ = _resample(method, params, X, y)
    assert _ir(y_bal) < _ir(y)


@pytest.mark.parametrize("method,params", ALL_METHOD_PARAMS)
def test_feature_count_unchanged(arrays, method, params):
    """Output must have the same number of feature columns as input."""
    X, y = arrays
    X_bal, _, _ = _resample(method, params, X, y)
    assert X_bal.shape[1] == X.shape[1]


@pytest.mark.parametrize("method,params", ALL_METHOD_PARAMS)
def test_reproducibility(arrays, method, params):
    X, y = arrays
    X1, y1, _ = _resample(method, params, X, y)
    X2, y2, _ = _resample(method, params, X, y)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


# ---------------------------------------------------------------------------
# _prepare
# ---------------------------------------------------------------------------

def test_prepare_row_count(df):
    X, y, *_ = _prepare(df, "label")
    assert len(y) == len(df)


def test_prepare_label_encoder(df):
    _, y_train, le, *_ = _prepare(df, "label")
    assert set(le.classes_) == {"majority", "minority"}
    assert set(np.unique(y_train)).issubset({0, 1})


def test_prepare_col_names(df):
    _, _, _, col_names, *_ = _prepare(df, "label")
    assert len(col_names) == 4  # f1–f4; label excluded


def test_prepare_does_not_modify_df(df):
    original_shape = df.shape
    _prepare(df, "label")
    assert df.shape == original_shape


def test_prepare_returns_encoder_for_categoricals(cat_df):
    """_prepare must return a fitted OrdinalEncoder when the df has categorical columns."""
    _, _, _, _, enc, cat_cols, *_ = _prepare(cat_df, "label")
    assert enc is not None
    assert "workclass" in cat_cols


def test_prepare_no_encoder_for_numeric_only(df):
    """_prepare must return enc=None when all feature columns are numeric."""
    _, _, _, _, enc, cat_cols, *_ = _prepare(df, "label")
    assert enc is None
    assert cat_cols == []


def test_unknown_method_raises(arrays):
    X, y = arrays
    with pytest.raises(ValueError, match="Unknown balancing method"):
        _resample("bogus", {}, X, y)


# ---------------------------------------------------------------------------
# SMOTE constraint: integer rounding and range clipping
# ---------------------------------------------------------------------------

@pytest.fixture
def integer_arrays():
    """
    Dataset whose every feature is a whole number (simulating columns like
    Pregnancies, Age, Outcome in the Pima diabetes dataset).
    60 majority (class 0), 20 minority (class 1).
    """
    rng = np.random.default_rng(7)
    X_maj = rng.integers(0, 10, size=(60, 3)).astype(float)
    X_min = rng.integers(0, 10, size=(20, 3)).astype(float)
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(60, dtype=int), np.ones(20, dtype=int)])
    return X, y


def test_smote_integer_columns_stay_integer(integer_arrays):
    """All synthetic SMOTE values must be whole numbers when original data is all integers."""
    X, y = integer_arrays
    X_bal, _, _ = _resample("smote", {"k_neighbors": 5}, X, y)
    synthetic = X_bal[len(X):]
    assert np.all(synthetic == np.floor(synthetic)), (
        "SMOTE produced fractional values in columns that were all integers in the original data"
    )


def test_smote_values_clamped_to_range(integer_arrays):
    """No SMOTE output value may fall outside the original per-column [min, max] range."""
    X, y = integer_arrays
    X_bal, _, _ = _resample("smote", {"k_neighbors": 5}, X, y)
    for j in range(X.shape[1]):
        col_min = X[:, j].min()
        col_max = X[:, j].max()
        assert X_bal[:, j].min() >= col_min, f"Column {j}: value below original minimum"
        assert X_bal[:, j].max() <= col_max, f"Column {j}: value above original maximum"


def test_combined_integer_columns_stay_integer(integer_arrays):
    """Combined (SMOTE + NearMiss) must also produce only whole numbers for integer columns."""
    X, y = integer_arrays
    X_bal, y_bal, log_info = _resample(
        "combined", {"k_neighbors": 5, "n_neighbors": 3}, X, y
    )
    # Check the added (SMOTE-generated) rows specifically
    added_X, _ = log_info["added"]
    assert np.all(added_X == np.floor(added_X)), (
        "Combined method produced fractional values in SMOTE-added rows for integer columns"
    )


# ---------------------------------------------------------------------------
# is_original flag in log_info
# ---------------------------------------------------------------------------

def test_smote_is_original_counts(arrays):
    """SMOTE: original rows flagged 1, synthetic rows flagged 0."""
    X, y = arrays
    X_bal, _, log_info = _resample("smote", {"k_neighbors": 5}, X, y)
    flags = log_info["is_original"]
    assert len(flags) == len(X_bal)
    assert int(flags.sum()) == len(X), "Expected exactly len(X) original rows"
    assert int((flags == 0).sum()) == len(X_bal) - len(X), "Expected synthetic count to match added rows"


def test_nearmiss_is_original_all_ones(arrays):
    """NearMiss only removes rows — every kept row must have is_original=1."""
    X, y = arrays
    X_bal, _, log_info = _resample("nearmiss", {"n_neighbors": 3}, X, y)
    flags = log_info["is_original"]
    assert len(flags) == len(X_bal)
    assert np.all(flags == 1), "NearMiss produced a row flagged as synthetic"


def test_combined_is_original_mixed(arrays):
    """Combined: some rows original (1), some synthetic (0); lengths match output."""
    X, y = arrays
    X_bal, _, log_info = _resample(
        "combined", {"k_neighbors": 5, "n_neighbors": 3}, X, y
    )
    flags = log_info["is_original"]
    assert len(flags) == len(X_bal)
    assert set(flags).issubset({0, 1}), "is_original must only contain 0 or 1"
    # Combined always produces some synthetic rows (SMOTE ran)
    assert (flags == 0).any(), "Combined should have at least some synthetic rows"
    assert (flags == 1).any(), "Combined should have at least some original rows"
