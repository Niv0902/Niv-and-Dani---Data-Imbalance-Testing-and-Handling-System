import threading
from typing import Dict, Optional, Any

_lock = threading.Lock()

# {dataset_id: {"df": DataFrame, "filename": str, "rows": int, "columns": int, "column_types": dict}}
_datasets: Dict[str, Dict[str, Any]] = {}

# {run_id: {"status": str, "stage": str, "progress_pct": int, "elapsed_sec": float,
#            "result": dict|None, "error": str|None, "cancelled": bool,
#            "dataset_id": str, "label_col": str}}
_runs: Dict[str, Dict[str, Any]] = {}


def store_dataset(dataset_id: str, data: Dict[str, Any]) -> None:
    with _lock:
        _datasets[dataset_id] = data


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _datasets.get(dataset_id)


def store_run(run_id: str, data: Dict[str, Any]) -> None:
    with _lock:
        _runs[run_id] = data


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _runs.get(run_id)


def update_run(run_id: str, updates: Dict[str, Any]) -> None:
    with _lock:
        if run_id in _runs:
            _runs[run_id].update(updates)


def get_all_runs() -> Dict[str, Dict[str, Any]]:
    with _lock:
        return dict(_runs)
