"""Upload and result persistence."""

from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

from agent.config import ALLOW_LOCAL_FILE_PATHS
from agent.schemas import ResultRef
from agent.services.mat_io import inspect_mat_file
from agent.services.paths import (
    RESULTS_INDEX_PATH,
    RUNS_ROOT,
    UPLOADS_INDEX_PATH,
    UPLOADS_ROOT,
    ensure_data_dirs,
)

_RESULTS_LOCK = threading.Lock()


def _read_index(path: Path) -> dict[str, Any]:
    ensure_data_dirs()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_index(path: Path, data: dict[str, Any]) -> None:
    ensure_data_dirs()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\s]+', "_", name.strip())
    return name.strip("._") or "file"


def make_run_dir(subgraph: str = "strain_estimation") -> str:
    ensure_data_dirs()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{sanitize_filename(subgraph)}"
    run_dir = RUNS_ROOT / run_id
    suffix = 2
    while run_dir.exists():
        run_dir = RUNS_ROOT / f"{run_id}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def register_upload(source_path: str | Path, original_name: str | None = None) -> dict[str, Any]:
    ensure_data_dirs()
    source = Path(source_path)
    if source.suffix.lower() != ".mat":
        raise ValueError("Only .mat files are supported.")
    if not source.exists():
        raise FileNotFoundError(source)

    file_id = uuid.uuid4().hex
    upload_dir = UPLOADS_ROOT / file_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(original_name or source.name)
    if not safe_name.lower().endswith(".mat"):
        safe_name += ".mat"
    target = upload_dir / safe_name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)

    inspection = inspect_mat_file(target)
    record = {
        "file_id": file_id,
        "original_name": original_name or source.name,
        "path": str(target),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **inspection,
    }
    index = _read_index(UPLOADS_INDEX_PATH)
    index[file_id] = record
    _write_index(UPLOADS_INDEX_PATH, index)
    return record


def resolve_file_reference(file_id: str | None = None, file_path: str | None = None) -> tuple[str | None, str]:
    if file_id:
        index = _read_index(UPLOADS_INDEX_PATH)
        record = index.get(file_id)
        if not record:
            raise ValueError(f"Unknown file_id: {file_id}")
        path = Path(record["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        return file_id, str(path)

    if file_path and ALLOW_LOCAL_FILE_PATHS:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return None, str(path)

    raise ValueError("Use file_id from the upload API. Local paths are disabled by default.")


def make_result_key(method: str, **params: Any) -> str:
    if method == "vector":
        return f"vector_g={params['g']}_Nx={params['Nx']}_Nz={params['Nz']}"
    if method == "cnn":
        return "cnn"
    if method == "bnn":
        return f"bnn_MC_test={params['MC_test']}"
    raise ValueError(f"Unknown result method: {method}")


def _make_result_base_name(run_dir: str, source_path: str, result_key: str) -> str:
    run_name = Path(run_dir).name
    file_name = sanitize_filename(Path(source_path).stem)
    safe_key = sanitize_filename(result_key)
    return f"{run_name}_{file_name}_{safe_key}"


def _register_result(ref: ResultRef) -> ResultRef:
    with _RESULTS_LOCK:
        index = _read_index(RESULTS_INDEX_PATH)
        index[ref["result_id"]] = dict(ref)
        _write_result_sidecar(ref)
        _write_index(RESULTS_INDEX_PATH, index)
    return ref


def _write_result_sidecar(ref: ResultRef) -> None:
    result_path = Path(ref["result_path"])
    sidecar_path = result_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(dict(ref), ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_result_path_is_safe(result_path: Path) -> Path:
    resolved = result_path.resolve()
    runs_root = RUNS_ROOT.resolve()
    if not resolved.is_file() or resolved.suffix.lower() != ".mat":
        raise ValueError("结果文件不存在或不是 .mat 文件。")
    if resolved != runs_root and runs_root not in resolved.parents:
        raise ValueError("只能恢复 data/runs 目录内的结果文件。")
    return resolved


def _infer_outputs_from_mat(result_path: Path) -> tuple[str, dict[str, dict[str, Any]], list[int]]:
    data = sio.loadmat(result_path)
    outputs: dict[str, dict[str, Any]] = {}
    for name, value in data.items():
        if name.startswith("__"):
            continue
        matrix = np.asarray(value)
        if matrix.ndim >= 2:
            outputs[name] = {"shape": list(matrix.shape)}
    if not outputs:
        raise ValueError("结果文件中没有可视化矩阵。")
    kind = "bnn" if "epistemic_uncertainty" in outputs else "array"
    first_shape = next(iter(outputs.values()))["shape"]
    return kind, outputs, first_shape


def recover_result(ref: dict[str, Any]) -> ResultRef:
    result_id = str(ref.get("result_id") or "")
    result_path_value = ref.get("result_path")
    if not result_id or not result_path_value:
        raise ValueError("恢复结果需要 result_id 和 result_path。")

    result_path = _ensure_result_path_is_safe(Path(str(result_path_value)))
    sidecar_path = result_path.with_suffix(".json")
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("result_id") != result_id:
            raise ValueError("结果 metadata 与 result_id 不匹配。")
        return _register_result(sidecar)

    kind, outputs, shape = _infer_outputs_from_mat(result_path)
    recovered: ResultRef = {
        "result_id": result_id,
        "file_id": str(ref.get("file_id") or ""),
        "file_path": str(ref.get("file_path") or ""),
        "result_key": str(ref.get("result_key") or result_path.stem),
        "result_path": str(result_path),
        "kind": kind,  # type: ignore[typeddict-item]
        "format": "mat",
        "shape": shape,
        "outputs": outputs,
    }
    return _register_result(recovered)


def save_array_result(
    run_dir: str,
    source_path: str,
    result_key: str,
    array: np.ndarray,
    file_id: str | None = None,
) -> ResultRef:
    result_id = uuid.uuid4().hex
    base_name = _make_result_base_name(run_dir, source_path, result_key)
    result_path = Path(run_dir) / f"{base_name}.mat"
    sio.savemat(result_path, {"strain": array})
    return _register_result(
        {
            "result_id": result_id,
            "file_id": file_id or "",
            "file_path": source_path,
            "result_key": result_key,
            "result_path": str(result_path),
            "kind": "array",
            "format": "mat",
            "shape": list(array.shape),
            "outputs": {"strain": {"shape": list(array.shape)}},
        }
    )


def save_bnn_result(
    run_dir: str,
    source_path: str,
    result_key: str,
    strain: np.ndarray,
    epistemic_uncertainty: np.ndarray,
    file_id: str | None = None,
) -> ResultRef:
    result_id = uuid.uuid4().hex
    base_name = _make_result_base_name(run_dir, source_path, result_key)
    result_path = Path(run_dir) / f"{base_name}.mat"
    sio.savemat(
        result_path,
        {
            "strain": strain,
            "epistemic_uncertainty": epistemic_uncertainty,
        },
    )
    return _register_result(
        {
            "result_id": result_id,
            "file_id": file_id or "",
            "file_path": source_path,
            "result_key": result_key,
            "result_path": str(result_path),
            "kind": "bnn",
            "format": "mat",
            "outputs": {
                "strain": {"shape": list(strain.shape)},
                "epistemic_uncertainty": {"shape": list(epistemic_uncertainty.shape)},
            },
        }
    )


def get_result(result_id: str) -> ResultRef:
    index = _read_index(RESULTS_INDEX_PATH)
    record = index.get(result_id)
    if not record:
        raise ValueError(f"Unknown result_id: {result_id}")
    return record


def load_result_array(result_id: str, name: str = "strain") -> dict[str, Any]:
    ref = get_result(result_id)
    data = sio.loadmat(ref["result_path"])
    if name not in data:
        available = [key for key in data if not key.startswith("__")]
        raise ValueError(f"Array {name} not found. Available arrays: {available}")
    matrix = np.asarray(data[name], dtype=float)
    finite = matrix[np.isfinite(matrix)]
    min_value = float(np.min(finite)) if finite.size else 0.0
    max_value = float(np.max(finite)) if finite.size else 0.0
    return {
        "result_id": result_id,
        "name": name,
        "shape": list(matrix.shape),
        "min": min_value,
        "max": max_value,
        "data": np.nan_to_num(matrix, nan=0.0).tolist(),
    }
