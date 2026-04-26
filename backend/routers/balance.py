import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict
from models import state
from services.balancing_service import start_pipeline

router = APIRouter()

VALID_METHODS = {"smote", "nearmiss", "combined"}


class BalanceRequest(BaseModel):
    dataset_id: str
    label_col: str
    method: str
    params: Dict[str, Any] = Field(default_factory=dict)
    held_out_size: float = Field(default=0.2, ge=0.05, le=0.5)


@router.post("/balance")
def balance(req: BalanceRequest):
    if req.method not in VALID_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown method '{req.method}'. Choose from: {', '.join(VALID_METHODS)}.",
        )

    ds = state.get_dataset(req.dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = ds["df"]
    if req.label_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.label_col}' not found.")

    run_id = str(uuid.uuid4())
    start_pipeline(run_id, df, req.label_col, req.method, req.params, req.held_out_size, req.dataset_id)
    return {"run_id": run_id}
