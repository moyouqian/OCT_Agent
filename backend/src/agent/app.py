"""FastAPI endpoints for uploads and result visualization."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agent.self_rag import (
    SELF_RAG_UPLOAD_DIR,
    SUPPORTED_KNOWLEDGE_EXTENSIONS,
    ingest_knowledge_file,
    knowledge_status,
)
from agent.services.storage import (
    get_result,
    load_result_array,
    recover_result,
    register_upload,
)

app = FastAPI(title="OCT Agent API")
_KNOWLEDGE_JOBS: dict[str, dict[str, Any]] = {}
_KNOWLEDGE_JOBS_LOCK = threading.Lock()

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


@app.get("/api/knowledge/status")
def get_knowledge_status() -> dict:
    return knowledge_status()


@app.post("/api/knowledge/upload")
def upload_knowledge_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_KNOWLEDGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported knowledge file type: {suffix}. Supported: {sorted(SUPPORTED_KNOWLEDGE_EXTENSIONS)}",
        )

    job_id = uuid.uuid4().hex
    safe_name = Path(file.filename).name
    target = SELF_RAG_UPLOAD_DIR / f"{job_id}_{safe_name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_bytes(file.file.read())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = {
        "job_id": job_id,
        "status": "queued",
        "filename": safe_name,
        "path": str(target),
        "result": None,
        "error": "",
    }
    with _KNOWLEDGE_JOBS_LOCK:
        _KNOWLEDGE_JOBS[job_id] = job
    background_tasks.add_task(_run_knowledge_ingestion_job, job_id, target)
    return job


@app.get("/api/knowledge/jobs/{job_id}")
def get_knowledge_job(job_id: str) -> dict:
    with _KNOWLEDGE_JOBS_LOCK:
        job = _KNOWLEDGE_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Knowledge job not found: {job_id}")
        return dict(job)


def _run_knowledge_ingestion_job(job_id: str, path: Path) -> None:
    _update_knowledge_job(job_id, status="running")
    try:
        result = ingest_knowledge_file(path)
    except Exception as exc:
        _update_knowledge_job(job_id, status="failed", error=str(exc))
        return
    _update_knowledge_job(job_id, status="succeeded", result=result)


def _update_knowledge_job(job_id: str, **updates: Any) -> None:
    with _KNOWLEDGE_JOBS_LOCK:
        job = _KNOWLEDGE_JOBS.get(job_id)
        if not job:
            return
        job.update(updates)


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
