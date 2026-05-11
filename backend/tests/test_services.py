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
from agent.graph import _research_messages, infer_route_from_text, supervisor
from agent.prompts import SUPERVISOR_PROMPT
from agent.research.graph import _needs_research_clarification, _parse_items
from agent.research.schemas import ResearchPlan
from agent.utils.structured import invoke_structured_json_schema
from langchain_core.messages import AIMessage, HumanMessage


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


def test_requested_deep_research_routes_without_llm():
    result = supervisor(
        {
            "messages": [HumanMessage(content="请调研 OCT angiography 最新进展")],
            "requested_sub_agent": "deep_research",
        }
    )

    assert result["sub_agent"] == "deep_research"


def test_route_keywords_cover_research_and_strain():
    assert infer_route_from_text("帮我做一份 OCT 文献综述") == "deep_research"
    assert infer_route_from_text("对 phase.mat 做 CNN 应变计算") == "strain_estimation"
    assert infer_route_from_text("你好") is None


def test_prompts_are_importable():
    assert "deep_research" in SUPERVISOR_PROMPT


def test_deep_research_context_trims_unrelated_history():
    messages = [
        HumanMessage(content="你好"),
        AIMessage(content="你好，我是 OCT Agent。"),
        HumanMessage(content="对 phase.mat 做应变计算"),
        AIMessage(content="已生成 1 个结果。"),
        HumanMessage(content="Deep Research: 介绍哈雷彗星"),
    ]

    trimmed = _research_messages({"messages": messages})

    assert trimmed == [messages[-1]]


def test_pending_deep_research_keeps_recent_clarification_context():
    messages = [
        HumanMessage(content="你好"),
        AIMessage(content="你好。"),
        HumanMessage(content="做个研究"),
        AIMessage(content="请问研究主题是什么？"),
        HumanMessage(content="研究哈雷彗星"),
    ]

    trimmed = _research_messages({"messages": messages, "research_pending": True})

    assert trimmed == messages[-4:]


def test_research_fallback_parsers_are_available():
    assert _needs_research_clarification("OCT") is True
    assert _needs_research_clarification("技术原理、最新临床应用进展、市场趋势") is False
    assert _parse_items("1. 技术原理\n2. 临床应用\n3. 市场趋势", max_items=2) == [
        "技术原理",
        "临床应用",
    ]


class _FakeStructuredModel:
    def __init__(self, result):
        self.result = result

    def invoke(self, messages):
        return self.result


class _FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def with_structured_output(self, schema, method, include_raw):
        self.calls.append((schema, method, include_raw))
        return _FakeStructuredModel(self.result)


def test_structured_helper_uses_json_schema_first():
    model = _FakeModel({"parsed": ResearchPlan(topics=["A"]), "parsing_error": None})

    parsed = invoke_structured_json_schema(model, ResearchPlan, [HumanMessage(content="plan")])

    assert parsed.topics == ["A"]
    assert model.calls[0] == (ResearchPlan, "json_schema", True)
