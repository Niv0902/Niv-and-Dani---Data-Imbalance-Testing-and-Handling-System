from fastapi import APIRouter, HTTPException
from models import state

router = APIRouter()


@router.get("/status/{run_id}")
def get_status(run_id: str):
    run = state.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return {
        "run_id": run_id,
        "status": run["status"],
        "stage": run["stage"],
        "progress_pct": run["progress_pct"],
        "elapsed_sec": run["elapsed_sec"],
        "error": run.get("error"),
    }


@router.post("/cancel/{run_id}")
def cancel_run(run_id: str):
    run = state.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    state.update_run(run_id, {"cancelled": True})
    return {"cancelled": True}
