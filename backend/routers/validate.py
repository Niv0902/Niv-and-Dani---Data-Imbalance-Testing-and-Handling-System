from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models import state
from services.validation_service import run_validation_checks, has_blocking_errors

router = APIRouter()


class ValidateRequest(BaseModel):
    dataset_id: str
    label_col: str


@router.post("/validate")
def validate(req: ValidateRequest):
    ds = state.get_dataset(req.dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    checks = run_validation_checks(ds["df"], req.label_col)
    return {
        "checks": checks,
        "has_errors": has_blocking_errors(checks),
        "passed": sum(1 for c in checks if c["status"] == "pass"),
        "warnings": sum(1 for c in checks if c["status"] == "warning"),
        "errors": sum(1 for c in checks if c["status"] == "error"),
    }
