import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import upload, dataset, validate, diagnosis, balance, status, results, export

app = FastAPI(
    title="ImbalanceKit API",
    description="Data Imbalance Testing and Handling System — Backend API",
    version="1.0.0",
)

_origins = os.environ.get("FRONTEND_URL", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

prefix = "/api"
app.include_router(upload.router, prefix=prefix, tags=["Upload"])
app.include_router(dataset.router, prefix=prefix, tags=["Dataset"])
app.include_router(validate.router, prefix=prefix, tags=["Validate"])
app.include_router(diagnosis.router, prefix=prefix, tags=["Diagnosis"])
app.include_router(balance.router, prefix=prefix, tags=["Balance"])
app.include_router(status.router, prefix=prefix, tags=["Status"])
app.include_router(results.router, prefix=prefix, tags=["Results"])
app.include_router(export.router, prefix=prefix, tags=["Export"])


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "ImbalanceKit API is running.",
        "allowed_origins": [o.strip() for o in _origins.split(",")],
    }
