from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.services.mat_io import load_single_matrix
from agent.services.models import gradient_to_strain
from agent.services.storage import (
    get_result,
    load_result_array,
    make_run_dir,
    register_upload,
    recover_result,
    save_array_result,
)
from agent.services.paths import RESULTS_INDEX_PATH
from agent.tools import compute_vector_strain


def test_load_single_matrix(tmp_path):
    mat_path = tmp_path / "phase.mat"
    sio.savemat(mat_path, {"phase": np.ones((8, 9))})

    name, matrix = load_single_matrix(mat_path)

    assert name == "phase"
    assert matrix.shape == (8, 9)


def test_upload_and_result_storage(tmp_path):
    mat_path = tmp_path / "phase.mat"
    sio.savemat(mat_path, {"phase": np.ones((8, 9))})

    upload = register_upload(mat_path, original_name="phase.mat")
    run_dir = make_run_dir("test")
    ref = save_array_result(
        run_dir=run_dir,
        source_path=upload["path"],
        result_key="vector_g=1_Nx=2_Nz=2",
        array=np.ones((3, 4)),
        file_id=upload["file_id"],
    )
    payload = load_result_array(ref["result_id"], "strain")

    assert upload["file_id"]
    assert payload["shape"] == [3, 4]
    assert payload["min"] == 1.0
    assert payload["max"] == 1.0


def test_gradient_to_strain_is_finite():
    phase = np.array([[0.0, np.nan], [1.0, -1.0]])
    strain = gradient_to_strain(phase)

    assert strain.shape == phase.shape
    assert np.isfinite(strain).all()


def test_vector_method_shape(tmp_path):
    mat_path = tmp_path / "phase.mat"
    y = np.linspace(0, 1, 16).reshape(16, 1)
    x = np.linspace(0, 1, 18).reshape(1, 18)
    sio.savemat(mat_path, {"phase": y + x})

    strain = compute_vector_strain(str(mat_path), Nx=3, Nz=3, g=1)

    assert strain.ndim == 2
    assert strain.shape[0] > 0
    assert strain.shape[1] > 0
    assert np.isfinite(strain).all()


def test_concurrent_result_saves_keep_all_index_entries():
    run_dir = make_run_dir("concurrent_test")

    def save_one(index: int):
        return save_array_result(
            run_dir=run_dir,
            source_path=f"phase_{index}.mat",
            result_key=f"vector_{index}",
            array=np.full((2, 3), index, dtype=float),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        refs = list(pool.map(save_one, range(4)))

    for ref in refs:
        assert get_result(ref["result_id"])["result_id"] == ref["result_id"]
        assert Path(ref["result_path"]).with_suffix(".json").exists()


def test_recover_result_restores_missing_index_entry():
    run_dir = make_run_dir("recover_test")
    ref = save_array_result(
        run_dir=run_dir,
        source_path="phase.mat",
        result_key="vector_recover",
        array=np.ones((2, 2), dtype=float),
    )

    index = json.loads(RESULTS_INDEX_PATH.read_text(encoding="utf-8"))
    index.pop(ref["result_id"])
    RESULTS_INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    restored = recover_result(dict(ref))
    payload = load_result_array(restored["result_id"], "strain")

    assert restored["result_id"] == ref["result_id"]
    assert payload["shape"] == [2, 2]


def test_recover_result_rejects_paths_outside_runs(tmp_path):
    outside = tmp_path / "outside.mat"
    sio.savemat(outside, {"strain": np.ones((2, 2), dtype=float)})

    try:
        recover_result(
            {
                "result_id": "outside",
                "result_path": str(outside),
                "result_key": "outside",
            }
        )
    except ValueError as exc:
        assert "data/runs" in str(exc)
    else:
        raise AssertionError("recover_result should reject paths outside data/runs")
