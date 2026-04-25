from typing import List, Dict, Any

import pandas as pd


def run_validation_checks(df: pd.DataFrame, label_col: str) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    if label_col not in df.columns:
        checks.append(_check("Label column exists", "error",
                             f"Column '{label_col}' was not found in the dataset.",
                             None))
        return checks

    label = df[label_col]
    feature_cols = [c for c in df.columns if c != label_col]

    # 1. No null labels
    null_count = int(label.isna().sum())
    if null_count > 0:
        checks.append(_check(
            "Label column: no null values", "error",
            f"Label column has {null_count} null value(s). These rows must be removed before continuing.",
            f"Found {null_count} null values in '{label_col}'.",
        ))
    else:
        checks.append(_check("Label column: no null values", "pass",
                             "No null values found in the label column.", None))

    # 2. At least 2 unique classes
    unique_count = int(label.dropna().nunique())
    if unique_count < 2:
        checks.append(_check(
            "Label column: at least 2 classes", "error",
            f"Label column must have at least 2 classes. Found: {unique_count}.",
            f"Unique values: {list(label.dropna().unique()[:10])}",
        ))
    else:
        checks.append(_check("Label column: at least 2 classes", "pass",
                             f"Found {unique_count} unique classes.", None))

    # 3. Each class must have at least 2 samples (required for stratified split)
    value_counts = label.dropna().value_counts()
    singletons = value_counts[value_counts < 2]
    if len(singletons) > 0:
        majority_are_singletons = len(singletons) > len(value_counts) * 0.5
        sample_names = ", ".join(f'"{v}"' for v in singletons.index[:5])
        if len(singletons) > 5:
            sample_names += f" and {len(singletons) - 5} more"
        if majority_are_singletons:
            msg = (
                f"{len(singletons)} of {len(value_counts)} classes have only 1 sample. "
                "This column looks like a numeric or ID column, not a class label. "
                "Go back and select a proper categorical label column."
            )
        else:
            msg = (
                f"{len(singletons)} class(es) have only 1 sample ({sample_names}). "
                "Each class needs at least 2 samples for train/test splitting."
            )
        checks.append(_check(
            "Label column: at least 2 samples per class", "error", msg,
            f"Classes with 1 sample: {sample_names}",
        ))
    else:
        checks.append(_check(
            "Label column: at least 2 samples per class", "pass",
            "All classes have at least 2 samples.", None,
        ))

    # 4. Missing values in feature columns
    if feature_cols:
        missing = df[feature_cols].isna().sum()
        cols_missing = missing[missing > 0]
        if len(cols_missing) > 0:
            total = int(cols_missing.sum())
            pct = round(total / (len(df) * len(feature_cols)) * 100, 1)
            details = "\n".join(
                f"  - {col}: {cnt} missing ({round(cnt / len(df) * 100, 1)}%)"
                for col, cnt in cols_missing.items()
            )
            checks.append(_check(
                "Feature columns: missing values", "warning",
                f"{len(cols_missing)} column(s) have missing values "
                f"({total} total, {pct}% of feature cells). "
                "The pipeline will impute these with column medians.",
                details,
            ))
        else:
            checks.append(_check("Feature columns: missing values", "pass",
                                 "No missing values in feature columns.", None))

    # 4. Minimum 50 rows (blocking)
    if len(df) < 50:
        checks.append(_check(
            "Dataset size: minimum 50 rows", "error",
            f"Dataset has only {len(df)} rows. At least 50 rows are required to run the pipeline.",
            "Upload a larger dataset with at least 50 samples.",
        ))
    else:
        checks.append(_check("Dataset size: minimum 50 rows", "pass",
                             f"Dataset has {len(df):,} rows.", None))

    # 5. No constant (zero-variance) feature columns
    if feature_cols:
        constant = [c for c in feature_cols if df[c].dropna().nunique() <= 1]
        if constant:
            checks.append(_check(
                "Feature columns: no constant columns", "warning",
                f"{len(constant)} constant column(s) found (zero variance). They will be ignored during balancing.",
                f"Columns: {', '.join(constant)}",
            ))
        else:
            checks.append(_check("Feature columns: no constant columns", "pass",
                                 "No constant columns found.", None))

    return checks


def has_blocking_errors(checks: List[Dict[str, Any]]) -> bool:
    return any(c["status"] == "error" for c in checks)


def _check(name: str, status: str, message: str, details) -> Dict[str, Any]:
    return {"name": name, "status": status, "message": message, "details": details}
