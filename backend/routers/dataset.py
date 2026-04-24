from fastapi import APIRouter, HTTPException, Query
from models import state

router = APIRouter()


def _get_dataset_or_404(dataset_id: str):
    ds = state.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return ds


@router.get("/dataset/columns")
def get_columns(dataset_id: str = Query(...)):
    ds = _get_dataset_or_404(dataset_id)
    df = ds["df"]
    col_types = ds["column_types"]
    return [
        {
            "name": col,
            "dtype": col_types.get(col, "unknown"),
            "unique_count": int(df[col].nunique()),
        }
        for col in df.columns
    ]


@router.get("/dataset/preview")
def get_preview(dataset_id: str = Query(...), rows: int = Query(default=5)):
    ds = _get_dataset_or_404(dataset_id)
    df = ds["df"]
    preview = df.head(rows).fillna("").astype(str)
    return {
        "columns": list(df.columns),
        "rows": preview.to_dict(orient="records"),
    }


@router.get("/dataset/column-summary")
def get_column_summary(dataset_id: str = Query(...), col: str = Query(...)):
    ds = _get_dataset_or_404(dataset_id)
    df = ds["df"]
    if col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{col}' not found.")
    series = df[col].dropna()
    total = len(series)
    counts = series.value_counts()
    classes = [
        {"name": str(name), "count": int(cnt), "pct": round(cnt / total * 100, 2)}
        for name, cnt in counts.items()
    ]
    return {"col": col, "total": total, "unique_count": len(counts), "classes": classes}
