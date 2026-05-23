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
from agent.tools import bnn_method, cnn_method, compute_vector_strain
from agent.graph import _research_messages, infer_route_from_text, self_rag_node, supervisor
from agent.prompts import SUPERVISOR_PROMPT
from agent.research.graph import _needs_research_clarification, _parse_items
from agent.research.schemas import ResearchPlan
from agent.self_rag import get_self_rag_config
from agent.utils.structured import invoke_structured_json_schema
from agent.app import app
from fastapi.testclient import TestClient
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


def test_compute_vector_strain_rejects_invalid_params(tmp_path):
    mat_path = tmp_path / "phase.mat"
    sio.savemat(mat_path, {"phase": np.ones((40, 40))})

    for bad_kwargs in ({"Nx": 0}, {"Nz": 0}, {"g": 0}):
        try:
            raised = False
            compute_vector_strain(str(mat_path), **bad_kwargs)
        except ValueError:
            raised = True
        assert raised, f"compute_vector_strain should reject {bad_kwargs}"


def test_strain_tools_return_error_text_when_no_file():
    # 无 file_id 且 file_ids 为空 -> 共享骨架应返回友好的中文错误文案而非抛异常。
    cnn_result = cnn_method.func(run_dir="", file_ids=[], physical_params={}, file_id="", file_path="")
    bnn_result = bnn_method.func(run_dir="", file_ids=[], physical_params={}, file_id="", file_path="")

    assert "CNN" in cnn_result and "错误" in cnn_result
    assert "BNN" in bnn_result and "错误" in bnn_result


def test_requested_deep_research_routes_without_llm():
    result = supervisor(
        {
            "messages": [HumanMessage(content="请调研 OCT angiography 最新进展")],
            "requested_sub_agent": "deep_research",
        }
    )

    assert result["sub_agent"] == "deep_research"


def test_requested_sub_agent_supports_all_subgraphs():
    for requested in ("strain_estimation", "self_rag", "chat"):
        result = supervisor(
            {
                "messages": [HumanMessage(content="随便一句话")],
                "requested_sub_agent": requested,
            }
        )
        assert result["sub_agent"] == requested


def test_route_keywords_cover_research_and_strain():
    assert infer_route_from_text("帮我做一份 OCT 文献综述") == "deep_research"
    assert infer_route_from_text("对 phase.mat 做 CNN 应变计算") == "strain_estimation"
    assert infer_route_from_text("你好") is None


def test_route_keywords_cover_new_self_rag_and_chinese_research():
    assert infer_route_from_text("请联网调研 OCT 最新进展") == "deep_research"
    assert infer_route_from_text("请基于本地知识库回答 phase unwrapping") == "self_rag"


def test_method_name_knowledge_question_skips_strain():
    # 方法名出现在知识问答里时，关键词层不应判定为应变计算（交由兜底 self_rag）。
    assert infer_route_from_text("介绍BNN") is None
    assert infer_route_from_text("BNN是什么") is None
    assert infer_route_from_text("CNN 和 BNN 有什么区别") is None
    # 出现明确计算动作词时仍走应变。
    assert infer_route_from_text("用 BNN 计算应变") == "strain_estimation"
    # 本轮已上传文件时，方法名/领域词触发应变计算。
    assert infer_route_from_text("矢量法应变", has_files=True) == "strain_estimation"
    # 明确知识问句即便带附件也不进应变。
    assert infer_route_from_text("介绍BNN", has_files=True) is None


def test_supervisor_routes_method_knowledge_question_to_self_rag():
    knowledge = supervisor({"messages": [HumanMessage(content="介绍BNN")]})
    assert knowledge["sub_agent"] == "self_rag"

    compute = supervisor({"messages": [HumanMessage(content="用 BNN 计算应变")]})
    assert compute["sub_agent"] == "strain_estimation"


def test_default_supervisor_routes_to_self_rag_without_llm():
    result = supervisor({"messages": [HumanMessage(content="你好，介绍一下 OCT")]})

    assert result["sub_agent"] == "self_rag"


def test_self_rag_falls_back_to_chat(monkeypatch):
    # 让闸门判定为"需要检索"，从而走到检索后兜底分支。
    monkeypatch.setattr("agent.graph.decide_retrieval", lambda messages: (True, {"decision": "retrieve"}))
    monkeypatch.setattr(
        "agent.graph.run_knowledge_query",
        lambda question: {"error": "empty index", "documents": [], "generation": "", "_used_chat_fallback": True},
    )
    monkeypatch.setattr("agent.graph.chat", lambda state: {"messages": [AIMessage(content="chat fallback")]})

    result = self_rag_node({"messages": [HumanMessage(content="普通问题")]})

    assert result["messages"][0].content == "chat fallback"
    assert result["self_rag_error"] == "empty index"


def test_retrieval_gate_short_circuits_empty_knowledge_base(monkeypatch):
    from agent import self_rag

    monkeypatch.setattr(self_rag, "knowledge_status", lambda: {"child_chunks": 0})

    should_retrieve, trace = self_rag.decide_retrieval([HumanMessage(content="你好")])

    assert should_retrieve is False
    assert trace["tier"] == "empty_kb"


def test_self_rag_node_gate_skips_retrieval(monkeypatch):
    monkeypatch.setattr(
        "agent.graph.decide_retrieval",
        lambda messages: (False, {"decision": "direct", "tier": "llm", "reason": "闲聊"}),
    )

    def _fail_if_called(question):
        raise AssertionError("run_knowledge_query should not run when gate routes to direct chat")

    monkeypatch.setattr("agent.graph.run_knowledge_query", _fail_if_called)
    monkeypatch.setattr("agent.graph.chat", lambda state: {"messages": [AIMessage(content="hi")]})

    result = self_rag_node({"messages": [HumanMessage(content="你好")]})

    assert result["messages"][0].content == "hi"
    assert result["self_rag_trace"]["gate"]["decision"] == "direct"
    assert result["self_rag_citations"] == []


def test_self_rag_config_uses_backend_internal_data_dir():
    config = get_self_rag_config()
    normalized_sqlite = config.sqlite_path.replace("\\", "/")
    normalized_chroma = config.chroma_dir.replace("\\", "/")

    assert "backend/data/self_rag" in normalized_sqlite
    assert normalized_chroma.endswith("backend/data/self_rag/chroma_store")


def test_knowledge_api_accepts_upload_and_exposes_job(monkeypatch):
    monkeypatch.setattr(
        "agent.app.ingest_knowledge_file",
        lambda path: {"skipped": 0, "total_parents": 1, "total_children": 1, "ingested_children": 1},
    )
    client = TestClient(app)

    response = client.post(
        "/api/knowledge/upload",
        files={"file": ("note.md", b"# Test\n\nKnowledge entry.", "text/markdown")},
    )

    assert response.status_code == 200
    payload = response.json()
    job = client.get(f"/api/knowledge/jobs/{payload['job_id']}").json()
    assert job["status"] == "succeeded"
    assert job["result"]["ingested_children"] == 1


def test_knowledge_api_rejects_unsupported_upload():
    client = TestClient(app)

    response = client.post(
        "/api/knowledge/upload",
        files={"file": ("image.png", b"not supported", "image/png")},
    )

    assert response.status_code == 400


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


_PROVIDER_ENV_VARS = [
    "SILICONFLOW_API_KEY",
    "SILICONFLOW_API_BASE",
    "SILICONFLOW_API_MODEL",
    "GROQ_API_KEY",
    "GROQ_API_BASE",
    "GROQ_API_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "MODEL",
]


def test_get_llm_selects_one_provider_group(monkeypatch):
    from agent import config

    for name in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # 仅设置 Groq 一组；SiliconFlow 的 base/model 故意不设，验证不会被串入。
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GROQ_API_BASE", "https://groq.example/v1")
    monkeypatch.setenv("GROQ_API_MODEL", "groq-model")
    config.get_llm.cache_clear()

    llm = config.get_llm()

    assert llm.model_name == "groq-model"
    assert str(llm.openai_api_base) == "https://groq.example/v1"
    config.get_llm.cache_clear()


def test_get_llm_raises_when_selected_provider_incomplete(monkeypatch):
    from agent import config

    for name in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # 选中 SiliconFlow（有 key），但缺 base/model -> 不应回退到其他 provider。
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sf-key")
    config.get_llm.cache_clear()

    try:
        with_error = False
        config.get_llm()
    except RuntimeError as exc:
        with_error = True
        assert "SiliconFlow" in str(exc)
    assert with_error, "incomplete provider config should raise RuntimeError"
    config.get_llm.cache_clear()


def test_summarize_conversation_skips_when_below_threshold():
    from agent.graph import summarize_conversation
    state = {"messages": [HumanMessage(content=f"msg {i}") for i in range(10)]}
    result = summarize_conversation(state)
    assert result == {}


def test_summarize_conversation_calls_llm_above_threshold(monkeypatch):
    from agent import graph as g
    from agent.graph import summarize_conversation

    monkeypatch.setattr(g, "get_llm", lambda: type("M", (), {
        "invoke": lambda self, msgs: type("R", (), {"content": "摘要内容"})()
    })())

    msgs = [HumanMessage(content=f"msg {i}") for i in range(25)]
    state = {"messages": msgs, "summary_message_count": 0, "conversation_summary": ""}
    result = summarize_conversation(state)

    assert result.get("conversation_summary") == "摘要内容"
    assert result.get("summary_message_count") == len(msgs) - 6


def test_summarize_conversation_skips_when_no_new_messages():
    from agent.graph import summarize_conversation
    msgs = [HumanMessage(content=f"msg {i}") for i in range(25)]
    state = {
        "messages": msgs,
        "summary_message_count": len(msgs) - 6,
        "conversation_summary": "已有摘要",
    }
    result = summarize_conversation(state)
    assert result == {}


def test_clear_summary_command_resets_state():
    from agent.graph import chat
    result = chat({
        "messages": [HumanMessage(content="清除摘要")],
        "conversation_summary": "旧摘要",
        "summary_message_count": 10,
    })
    assert result["conversation_summary"] == ""
    assert result["summary_message_count"] == 0
    assert "清除" in result["messages"][0].content
