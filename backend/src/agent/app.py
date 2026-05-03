"""FastAPI endpoints for uploads and result visualization."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agent.services.storage import (
    get_result,
    load_result_array,
    recover_result,
    register_upload,
)

app = FastAPI(title="OCT Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/files/upload")
def upload_mat_file(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".mat"):
        raise HTTPException(status_code=400, detail="Only .mat files are supported.")

    try:
        return _register_uploaded_bytes(file.file.read(), file.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _register_uploaded_bytes(contents: bytes, filename: str) -> dict:
    with NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)
    try:
        return register_upload(tmp_path, original_name=filename)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/api/results/{result_id}/array")
def result_array(result_id: str, name: str = "strain") -> dict:
    try:
        return load_result_array(result_id, name=name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/results/{result_id}/metadata")
def result_metadata(result_id: str) -> dict:
    try:
        return get_result(result_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/results/recover")
def recover_result_metadata(ref: dict) -> dict:
    try:
        return recover_result(ref)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/results/{result_id}/download")
def download_result(result_id: str) -> FileResponse:
    try:
        ref = get_result(result_id)
        path = Path(ref["result_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
