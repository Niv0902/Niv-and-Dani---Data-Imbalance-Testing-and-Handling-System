from fastapi import APIRouter, HTTPException, Query
from models import state
from services.diagnosis_service import compute_diagnosis

router = APIRouter()


@router.get("/diagnosis")
def get_diagnosis(dataset_id: str = Query(...), label_col: str = Query(...)):
    ds = state.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    df = ds["df"]
    if label_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{label_col}' not found.")
    return compute_diagnosis(df, label_col)
