from fastapi import APIRouter, UploadFile, File, HTTPException
from models import state
from services.file_service import parse_upload

router = APIRouter()


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    dataset_id, metadata = await parse_upload(file)
    state.store_dataset(dataset_id, metadata)
    return {
        "dataset_id": dataset_id,
        "filename": metadata["filename"],
        "rows": metadata["rows"],
        "columns": metadata["columns"],
        "column_types": metadata["column_types"],
    }
