import threading
import time
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

from config import RANDOM_STATE
from models import state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_pipeline(
    run_id: str,
    df: pd.DataFrame,
    label_col: str,
    method: str,
    params: Dict[str, Any],
    dataset_id: str = "",
) -> None:
    state.store_run(run_id, {
        "status": "running",
        "stage": "Queued",
        "progress_pct": 0,
        "elapsed_sec": 0.0,
        "result": None,
        "error": None,
        "cancelled": False,
        "dataset_id": dataset_id,
    })
    thread = threading.Thread(
        target=_pipeline,
        args=(run_id, df, label_col, method, params),
        daemon=True,
    )
    thread.start()


# ---------------------------------------------------------------------------
# Internal pipeline
# ---------------------------------------------------------------------------

def _update(run_id: str, stage: str, pct: int, start: float) -> None:
    run = state.get_run(run_id)
    if run and run.get("cancelled"):
        raise InterruptedError("Pipeline cancelled by user.")
    state.update_run(run_id, {
        "stage": stage,
        "progress_pct": pct,
        "elapsed_sec": round(time.time() - start, 1),
    })


def _pipeline(
    run_id: str,
    df: pd.DataFrame,
    label_col: str,
    method: str,
    params: Dict[str, Any],
) -> None:
    start = time.time()
    try:
        _update(run_id, "Preparing data", 20, start)
        X, y, le, col_names, enc, cat_cols, numeric_cols = _prepare(df, label_col)

        _update(run_id, "Applying resampling", 60, start)
        X_bal, y_bal, log_info = _resample(method, params, X, y)

        _update(run_id, "Computing statistics", 90, start)

        ir_before = _ir(y)
        ir_after  = _ir(y_bal)

        def _dist(y_arr: np.ndarray) -> Dict[str, int]:
            counts = np.bincount(y_arr, minlength=len(le.classes_))
            return {str(le.classes_[i]): int(counts[i]) for i in range(len(le.classes_))}

        dist_before = _dist(y)
        dist_after  = _dist(y_bal)

        is_original = log_info["is_original"]
        balanced_df = _decode_df(X_bal, y_bal, col_names, le, label_col, enc, cat_cols, numeric_cols)
        balanced_df["is_original"] = is_original

        log_parts = []
        if log_info.get("added") is not None:
            X_add, y_add = log_info["added"]
            if len(X_add) > 0:
                df_add = _decode_df(X_add, y_add, col_names, le, label_col, enc, cat_cols, numeric_cols)
                df_add.insert(0, "change_type", "added")
                log_parts.append(df_add)
        if log_info.get("deleted") is not None:
            X_del, y_del = log_info["deleted"]
            if len(X_del) > 0:
                df_del = _decode_df(X_del, y_del, col_names, le, label_col, enc, cat_cols, numeric_cols)
                df_del.insert(0, "change_type", "deleted")
                log_parts.append(df_del)
        log_df = pd.concat(log_parts, ignore_index=True) if log_parts else None

        elapsed = round(time.time() - start, 2)

        state.update_run(run_id, {
            "status": "completed",
            "stage": "Done",
            "progress_pct": 100,
            "elapsed_sec": elapsed,
            "result": {
                "run_id": run_id,
                "method": method,
                "params": params,
                "ir_before": ir_before,
                "ir_after": ir_after,
                "class_distribution_before": dist_before,
                "class_distribution_after": dist_after,
                "total_before": int(len(y)),
                "total_after": int(len(y_bal)),
                "elapsed_seconds": elapsed,
                "class_names": [str(c) for c in le.classes_.tolist()],
                "original_count": int(is_original.sum()),
                "synthetic_count": int((is_original == 0).sum()),
                "balanced_df": balanced_df,
                "log_df": log_df,
                "label_col": label_col,
            },
        })

    except InterruptedError:
        state.update_run(run_id, {"status": "cancelled", "stage": "Cancelled"})
    except Exception as exc:
        state.update_run(run_id, {
            "status": "error",
            "stage": "Error",
            "error": str(exc),
            "elapsed_sec": round(time.time() - start, 1),
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare(
    df: pd.DataFrame,
    label_col: str,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, List[str]]:
    df_clean = df.dropna(subset=[label_col]).copy()
    feature_cols = [c for c in df_clean.columns if c != label_col]

    X = df_clean[feature_cols]
    y_raw = df_clean[label_col]

    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))

    counts = np.bincount(y)
    too_few = [
        f'"{le.classes_[i]}" ({int(counts[i])} sample)'
        for i in range(len(counts))
        if counts[i] < 2
    ]
    if too_few:
        many_singletons = len(too_few) > len(counts) * 0.5
        suggestion = (
            "This column may not be a proper label column — go back to Column Selection "
            "and choose a categorical column instead."
            if many_singletons else
            "Add more data for these classes, or go back and select a different label column."
        )
        raise ValueError(
            f"{len(too_few)} class(es) have fewer than 2 samples: "
            f"{', '.join(too_few[:5])}"
            f"{' …' if len(too_few) > 5 else ''}. "
            f"{suggestion}"
        )

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = X.select_dtypes(exclude=[np.number]).columns.tolist()
    col_names    = numeric_cols + [f"enc_{c}" for c in cat_cols]

    enc = None
    parts = []
    if numeric_cols:
        parts.append(X[numeric_cols].values.astype(float))
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        parts.append(enc.fit_transform(X[cat_cols].astype(str)))

    X_encoded = np.hstack(parts) if parts else np.empty((len(X), 0))
    X_imputed = SimpleImputer(strategy="median").fit_transform(X_encoded)

    return X_imputed, y, le, col_names, enc, cat_cols, numeric_cols


def _decode_df(
    X: np.ndarray, y: np.ndarray,
    col_names: List[str], le: LabelEncoder, label_col: str,
    enc, cat_cols: List[str], numeric_cols: List[str],
) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=col_names)
    df[label_col] = le.inverse_transform(y)
    if enc is not None:
        n_num = len(numeric_cols)
        cat_part = X[:, n_num:]
        cat_int = np.empty(cat_part.shape, dtype=int)
        for i, cats in enumerate(enc.categories_):
            cat_int[:, i] = np.clip(np.round(cat_part[:, i]).astype(int), 0, len(cats) - 1)
        decoded = enc.inverse_transform(cat_int)
        for i, col in enumerate(cat_cols):
            df[col] = decoded[:, i]
            df.drop(columns=[f"enc_{col}"], inplace=True)
        df = df[numeric_cols + cat_cols + [label_col]]
    return df


def _constrain_to_original(X_bal: np.ndarray, X_orig: np.ndarray) -> np.ndarray:
    """
    Per-column post-processing for SMOTE synthetic rows:
      1. Clip to [col_min, col_max] of the original data.
      2. Round to integer if every original value in that column is a whole number
         (e.g. Pregnancies, Age, Outcome).
    Applying to all rows is safe — original rows are already in-range and unaffected.
    """
    result = X_bal.copy()
    for j in range(X_orig.shape[1]):
        col_min = float(X_orig[:, j].min())
        col_max = float(X_orig[:, j].max())
        result[:, j] = np.clip(result[:, j], col_min, col_max)
        if np.all(X_orig[:, j] == np.floor(X_orig[:, j])):
            result[:, j] = np.round(result[:, j])
    return result


def _resample(
    method: str, params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, dict]:
    if method == "smote":
        k = int(params.get("k_neighbors", 5))
        minority_count = int(np.bincount(y_train).min())
        k = min(k, minority_count - 1)
        if k < 1:
            k = 1
        smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)
        X_bal = _constrain_to_original(X_bal, X_train)
        n = len(X_train)
        is_original = np.concatenate([np.ones(n, dtype=int), np.zeros(len(X_bal) - n, dtype=int)])
        return X_bal, y_bal, {"added": (X_bal[n:], y_bal[n:]), "deleted": None, "is_original": is_original}

    elif method == "nearmiss":
        version = int(params.get("version", 1))
        n = int(params.get("n_neighbors", 3))
        nm = NearMiss(version=version, n_neighbors=n)
        X_bal, y_bal = nm.fit_resample(X_train, y_train)
        kept = set(nm.sample_indices_)
        deleted_mask = np.array([i not in kept for i in range(len(X_train))])
        is_original = np.ones(len(X_bal), dtype=int)
        return X_bal, y_bal, {"added": None, "deleted": (X_train[deleted_mask], y_train[deleted_mask]), "is_original": is_original}

    elif method == "combined":
        k = int(params.get("k_neighbors", 5))
        minority_count = int(np.bincount(y_train).min())
        k = min(k, minority_count - 1)
        if k < 1:
            k = 1
        version = int(params.get("nearmiss_version", 1))
        n = int(params.get("n_neighbors", 3))
        smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
        X_s, y_s = smote.fit_resample(X_train, y_train)
        X_s = _constrain_to_original(X_s, X_train)  # constrain before NearMiss uses distances
        n_orig = len(X_train)
        added_X, added_y = X_s[n_orig:], y_s[n_orig:]
        nm = NearMiss(version=version, n_neighbors=n)
        X_bal, y_bal = nm.fit_resample(X_s, y_s)
        kept = set(nm.sample_indices_)
        deleted_mask = np.array([i not in kept for i in range(len(X_s))])
        is_orig_s = np.concatenate([np.ones(n_orig, dtype=int), np.zeros(len(X_s) - n_orig, dtype=int)])
        is_original = is_orig_s[nm.sample_indices_]
        return X_bal, y_bal, {"added": (added_X, added_y), "deleted": (X_s[deleted_mask], y_s[deleted_mask]), "is_original": is_original}

    raise ValueError(f"Unknown balancing method: {method!r}")


def _ir(y: np.ndarray) -> float:
    counts = np.bincount(y)
    if counts.min() == 0:
        return 9999.0
    return round(float(counts.max() / counts.min()), 4)
