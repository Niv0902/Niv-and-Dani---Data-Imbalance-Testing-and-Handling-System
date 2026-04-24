from typing import Dict, Any, List

import pandas as pd


def compute_diagnosis(df: pd.DataFrame, label_col: str) -> Dict[str, Any]:
    label = df[label_col].dropna()
    value_counts = label.value_counts()
    total = int(len(label))

    classes: List[Dict[str, Any]] = [
        {"name": str(name), "count": int(cnt), "pct": round(cnt / total * 100, 2)}
        for name, cnt in value_counts.items()
    ]

    majority_count = int(value_counts.iloc[0])
    minority_count = int(value_counts.iloc[-1])
    ir = round(majority_count / minority_count, 4) if minority_count > 0 else 9999.0

    if ir < 3:
        severity = "Low"
    elif ir < 10:
        severity = "Medium"
    elif ir < 50:
        severity = "High"
    else:
        severity = "Extreme"

    return {
        "ir": ir,
        "severity": severity,
        "classes": classes,
        "majority_class": classes[0]["name"],
        "minority_class": classes[-1]["name"],
        "total_samples": total,
    }
