# ImbalanceKit — Developer Guide

## Project Overview
A 9-screen wizard web app for diagnosing class imbalance, applying balancing methods
(SMOTE / NearMiss / Combined), and comparing before/after model performance.

Capstone Project 26-1-D-3 — Braude College of Engineering  
Authors: Niv Oren & Daniel Levovsky | Advisor: Dr. Avital Shulner Tal

## Stack
- **Backend**: Python 3.13 + FastAPI + pandas + scikit-learn + imbalanced-learn
- **Frontend**: React + Vite + Recharts + react-router-dom + axios
- **API**: REST/JSON at `http://localhost:8000/api`
- **Swagger docs**: `http://localhost:8000/docs`

## How to Run

### Backend (port 8000)
```
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Or double-click `start_backend.bat`

### Frontend (port 5173)
```
cd frontend
npm run dev
```
Or double-click `start_frontend.bat`

Then open http://localhost:5173

## Architecture

```
backend/
  main.py                # FastAPI app, CORS, router registration
  config.py              # Constants (MAX_FILE_SIZE, RANDOM_STATE, etc.)
  models/state.py        # Thread-safe in-memory store for datasets and runs
  services/
    file_service.py        # Upload parsing (CSV/XLSX, auto-delimiter detection)
    validation_service.py  # 5 data quality checks (blocking errors vs warnings)
    diagnosis_service.py   # IR computation and class distribution
    balancing_service.py   # Pipeline: split -> resample -> train -> evaluate
  routers/
    upload.py / dataset.py / validate.py / diagnosis.py
    balance.py / status.py / results.py / export.py

frontend/src/
  api/client.js          # Axios wrappers for all 12 endpoints
  context/AppContext.jsx # Global wizard state (datasetId, labelCol, runs, etc.)
  pages/                 # 9 pages, one per wizard step
  components/            # StepIndicator, ConfusionMatrix
```

## Key Design Decisions
- **No data leakage**: balancing applied ONLY to training split, never to test set
- **Deterministic**: RANDOM_STATE=42 on all split/model calls
- **In-memory state**: datasets and runs stored in thread-safe dicts in `models/state.py`
- **Background threading**: balancing pipeline runs in a daemon thread, polled every 2s
- **Model**: RandomForestClassifier (100 trees) — robust to mixed/encoded features
- **Feature encoding**: OrdinalEncoder for categoricals + median imputation before SMOTE
- **SMOTE k-guard**: k_neighbors auto-clamped to minority_count-1 to prevent crashes

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | /api/upload | Upload CSV/XLSX dataset |
| GET | /api/dataset/columns | List columns with types |
| GET | /api/dataset/preview | First N rows |
| GET | /api/dataset/column-summary | Class counts for a column |
| POST | /api/validate | Run 5 quality checks |
| GET | /api/diagnosis | IR + class distribution |
| POST | /api/balance | Start balancing pipeline |
| GET | /api/status/{run_id} | Poll pipeline progress |
| POST | /api/cancel/{run_id} | Cancel running pipeline |
| GET | /api/results/{run_id} | Full before/after metrics |
| GET | /api/export/dataset/{run_id} | Download balanced CSV |
| GET | /api/export/summary/{run_id} | Download run summary JSON |
| GET | /api/export/all/{run_id} | Download ZIP of both |
