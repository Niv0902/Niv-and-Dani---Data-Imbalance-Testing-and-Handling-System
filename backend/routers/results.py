from fastapi import APIRouter, HTTPException
from models import state

router = APIRouter()


@router.get("/results/{run_id}")
def get_results(run_id: str):
    run = state.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    if run["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Run is not completed yet. Status: {run['status']}")

    result = run["result"]
    # Exclude non-serialisable fields stored for export use
    return {k: v for k, v in result.items() if k not in ("balanced_df", "le")}
