import threading
import time
import uuid
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

from config import RANDOM_STATE, DEFAULT_N_ESTIMATORS
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
    test_size: float,
) -> None:
    state.store_run(run_id, {
        "status": "running",
        "stage": "Queued",
        "progress_pct": 0,
        "elapsed_sec": 0.0,
        "result": None,
        "error": None,
        "cancelled": False,
    })
    thread = threading.Thread(
        target=_pipeline,
        args=(run_id, df, label_col, method, params, test_size),
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
    test_size: float,
) -> None:
    start = time.time()
    try:
        _update(run_id, "Splitting data", 10, start)
        X_train, X_test, y_train, y_test, le, col_names = _prepare(
            df, label_col, test_size
        )

        _update(run_id, "Training baseline model", 25, start)
        clf_base = RandomForestClassifier(
            n_estimators=DEFAULT_N_ESTIMATORS, random_state=RANDOM_STATE
        )
        clf_base.fit(X_train, y_train)
        y_pred_before = clf_base.predict(X_test)

        _update(run_id, "Applying resampling", 50, start)
        X_bal, y_bal = _resample(method, params, X_train, y_train)

        _update(run_id, "Training balanced model", 70, start)
        clf_bal = RandomForestClassifier(
            n_estimators=DEFAULT_N_ESTIMATORS, random_state=RANDOM_STATE
        )
        clf_bal.fit(X_bal, y_bal)
        y_pred_after = clf_bal.predict(X_test)

        _update(run_id, "Computing metrics", 90, start)
        metrics_before = _metrics(y_test, y_pred_before, le)
        metrics_after = _metrics(y_test, y_pred_after, le)

        ir_before = _ir(y_train)
        ir_after = _ir(y_bal)

        def _dist(y_arr: np.ndarray) -> Dict[str, int]:
            counts = np.bincount(y_arr, minlength=len(le.classes_))
            return {str(le.classes_[i]): int(counts[i]) for i in range(len(le.classes_))}

        # Build exportable balanced DataFrame
        balanced_df = pd.DataFrame(X_bal, columns=col_names)
        balanced_df[label_col] = le.inverse_transform(y_bal)

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
                "test_size": test_size,
                "ir_before": ir_before,
                "ir_after": ir_after,
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
                "confusion_matrix_before": confusion_matrix(y_test, y_pred_before).tolist(),
                "confusion_matrix_after": confusion_matrix(y_test, y_pred_after).tolist(),
                "class_distribution_before": _dist(y_train),
                "class_distribution_after": _dist(y_bal),
                "elapsed_seconds": elapsed,
                "class_names": [str(c) for c in le.classes_.tolist()],
                "balanced_df": balanced_df,
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
    test_size: float,
) -> Tuple:
    df_clean = df.dropna(subset=[label_col]).copy()
    feature_cols = [c for c in df_clean.columns if c != label_col]

    X = df_clean[feature_cols]
    y_raw = df_clean[label_col]

    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))

    # Pre-flight: every class must have at least 2 samples before we even attempt a split.
    pre_counts = np.bincount(y)
    too_few_pre = [
        f'"{le.classes_[i]}" ({int(pre_counts[i])} sample)'
        for i in range(len(pre_counts))
        if pre_counts[i] < 2
    ]
    if too_few_pre:
        many_singletons = len(too_few_pre) > len(pre_counts) * 0.5
        suggestion = (
            "This column may not be a proper label column — go back to Column Selection "
            "and choose a categorical column instead."
            if many_singletons else
            "Add more data for these classes, or go back and select a different label column."
        )
        raise ValueError(
            f"{len(too_few_pre)} class(es) have fewer than 2 samples: "
            f"{', '.join(too_few_pre[:5])}"
            f"{' …' if len(too_few_pre) > 5 else ''}. "
            f"{suggestion}"
        )

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    col_names = numeric_cols + [f"enc_{c}" for c in cat_cols]

    parts = []
    if numeric_cols:
        parts.append(X[numeric_cols].values.astype(float))
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        parts.append(enc.fit_transform(X[cat_cols].astype(str)))

    X_encoded = np.hstack(parts) if parts else np.empty((len(X), 0))

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Post-split: a class could lose all its training samples with a large test_size.
    post_counts = np.bincount(y_train, minlength=len(le.classes_))
    too_few_post = [
        f'"{le.classes_[i]}" ({int(post_counts[i])} sample)'
        for i in range(len(post_counts))
        if post_counts[i] < 2
    ]
    if too_few_post:
        raise ValueError(
            f"After the train/test split, the following class(es) have fewer than 2 training "
            f"samples: {', '.join(too_few_post)}. "
            f"The current test size is {int(test_size * 100)}%. "
            "Try reducing the test size (e.g. to 10%) or add more samples for these classes."
        )

    return X_train, X_test, y_train, y_test, le, col_names


def _resample(
    method: str, params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "smote":
        k = int(params.get("k_neighbors", 5))
        minority_count = int(np.bincount(y_train).min())
        k = min(k, minority_count - 1)
        if k < 1:
            k = 1
        return SMOTE(k_neighbors=k, random_state=RANDOM_STATE).fit_resample(X_train, y_train)

    elif method == "nearmiss":
        version = int(params.get("version", 1))
        n = int(params.get("n_neighbors", 3))
        return NearMiss(version=version, n_neighbors=n).fit_resample(X_train, y_train)

    elif method == "combined":
        k = int(params.get("k_neighbors", 5))
        minority_count = int(np.bincount(y_train).min())
        k = min(k, minority_count - 1)
        if k < 1:
            k = 1
        version = int(params.get("nearmiss_version", 1))
        n = int(params.get("n_neighbors", 3))
        X_s, y_s = SMOTE(k_neighbors=k, random_state=RANDOM_STATE).fit_resample(X_train, y_train)
        return NearMiss(version=version, n_neighbors=n).fit_resample(X_s, y_s)

    raise ValueError(f"Unknown balancing method: {method!r}")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, le: LabelEncoder) -> Dict[str, Any]:
    class_names = le.classes_.tolist()
    counts = np.bincount(y_true, minlength=len(class_names))
    minority_idx = int(counts.argmin())

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    per_class = []
    for i, name in enumerate(class_names):
        key = str(i)
        r = report.get(key, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        per_class.append({
            "class_name": str(name),
            "precision": round(float(r["precision"]), 4),
            "recall": round(float(r["recall"]), 4),
            "f1": round(float(r["f1-score"]), 4),
            "support": int(r["support"]),
        })

    m = report.get(str(minority_idx), {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "minority_recall": round(float(m["recall"]), 4),
        "minority_precision": round(float(m["precision"]), 4),
        "minority_f1": round(float(m["f1-score"]), 4),
        "per_class": per_class,
    }


def _ir(y: np.ndarray) -> float:
    counts = np.bincount(y)
    if counts.min() == 0:
        return 9999.0
    return round(float(counts.max() / counts.min()), 4)
