import io
import json
import zipfile
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models import state

router = APIRouter()


def _get_completed_run(run_id: str):
    run = state.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    if run["status"] != "completed":
        raise HTTPException(status_code=409, detail="Run is not completed yet.")
    return run


@router.get("/export/dataset/{run_id}")
def export_dataset(run_id: str):
    run = _get_completed_run(run_id)
    result = run["result"]
    df = result["balanced_df"]

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    filename = f"balanced_dataset_{run_id[:8]}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/summary/{run_id}")
def export_summary(run_id: str):
    run = _get_completed_run(run_id)
    result = run["result"]

    summary = {k: v for k, v in result.items() if k not in ("balanced_df", "le")}
    summary["exported_at"] = datetime.now(timezone.utc).isoformat()

    buf = io.StringIO()
    json.dump(summary, buf, indent=2, default=str)
    buf.seek(0)

    filename = f"run_summary_{run_id[:8]}.json"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/all/{run_id}")
def export_all(run_id: str):
    run = _get_completed_run(run_id)
    result = run["result"]

    # Build CSV bytes
    csv_buf = io.StringIO()
    result["balanced_df"].to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode()

    # Build JSON bytes
    summary = {k: v for k, v in result.items() if k not in ("balanced_df", "le")}
    summary["exported_at"] = datetime.now(timezone.utc).isoformat()
    json_bytes = json.dumps(summary, indent=2, default=str).encode()

    # Pack into ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"balanced_dataset_{run_id[:8]}.csv", csv_bytes)
        zf.writestr(f"run_summary_{run_id[:8]}.json", json_bytes)
    zip_buf.seek(0)

    filename = f"imbalancekit_export_{run_id[:8]}.zip"
    return StreamingResponse(
        iter([zip_buf.read()]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
