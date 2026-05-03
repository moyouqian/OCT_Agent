"""Filesystem paths used by the backend."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = PROJECT_ROOT / "backend"
DATA_ROOT = PROJECT_ROOT / "data"
UPLOADS_ROOT = DATA_ROOT / "uploads"
RUNS_ROOT = DATA_ROOT / "runs"
RESULTS_INDEX_PATH = DATA_ROOT / "results.json"
UPLOADS_INDEX_PATH = DATA_ROOT / "uploads.json"
ASSETS_ROOT = PROJECT_ROOT / "assets"
DEFAULT_CNN_MODEL_PATH = ASSETS_ROOT / "cnn" / "model.pth"
DEFAULT_BNN_MODEL_PATH = ASSETS_ROOT / "bnn" / "model.pth"


def ensure_data_dirs() -> None:
    for path in (DATA_ROOT, UPLOADS_ROOT, RUNS_ROOT):
        path.mkdir(parents=True, exist_ok=True)

