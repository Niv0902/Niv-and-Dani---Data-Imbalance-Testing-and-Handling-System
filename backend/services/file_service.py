import io
import uuid
from typing import Tuple, Dict, Any

import pandas as pd
from fastapi import UploadFile, HTTPException

from config import MAX_FILE_SIZE_BYTES


def _detect_dtype(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


async def parse_upload(file: UploadFile) -> Tuple[str, Dict[str, Any]]:
    content = await file.read()

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="File exceeds the 50 MB limit. Please reduce the dataset size and try again.",
        )

    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "csv":
        df = _parse_csv(content, filename)
    elif ext in ("xlsx", "xls"):
        df = _parse_excel(content, filename)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a CSV or XLSX file.",
        )

    dataset_id = str(uuid.uuid4())
    column_types = {col: _detect_dtype(df[col]) for col in df.columns}

    return dataset_id, {
        "df": df,
        "filename": filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_types": column_types,
        "dataset_id": dataset_id,
    }


def _parse_csv(content: bytes, filename: str) -> pd.DataFrame:
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep)
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")


def _parse_excel(content: bytes, filename: str) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(content), sheet_name=0)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse XLSX: {exc}")
