"""Adapters for the built-in Self-RAG knowledge base."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, get_buffer_string
from pydantic import BaseModel, Field

from agent.config import get_llm
from agent.prompts import build_retrieval_gate_prompt
from agent.utils.structured import invoke_structured_json_schema
from self_rag_engine.config import DEFAULT_SELF_RAG_DATA_DIR, SelfRagConfig
from self_rag_engine.graph import build_self_rag_graph, initial_state
from self_rag_engine.ingestion import ingest_file
from self_rag_engine.llm import build_chat_backend
from self_rag_engine.retrieval import HybridRetriever


load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env")

SUPPORTED_KNOWLEDGE_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}
SELF_RAG_DATA_DIR = DEFAULT_SELF_RAG_DATA_DIR
SELF_RAG_UPLOAD_DIR = SELF_RAG_DATA_DIR / "uploads"

# 本地知识库的领域描述，供检索闸门判断 query 是否落在库内。
KNOWLEDGE_DOMAIN = (
    "OCT（光学相干断层成像）应变估计相关：相位解包裹、矢量法 / CNN / BNN 应变计算、"
    "OCT 成像原理、弹性成像，以及已索引的相关论文与笔记。"
)
# 送入闸门分类器的最近对话轮数（覆盖依赖上文的追问）。
GATE_CONTEXT_TURNS = 6

# 编译好的 Self-RAG 图单例，首次请求时按当前 config 构建，后续复用。
# HybridRetriever 初始化需要加载 ChromaDB + BM25 索引，重建代价较高。
_SELF_RAG_GRAPH = None
_SELF_RAG_GRAPH_LOCK = threading.Lock()


def _get_self_rag_graph():
    global _SELF_RAG_GRAPH
    if _SELF_RAG_GRAPH is None:
        with _SELF_RAG_GRAPH_LOCK:
            if _SELF_RAG_GRAPH is None:
                _SELF_RAG_GRAPH = build_self_rag_graph(config=get_self_rag_config())
    return _SELF_RAG_GRAPH


class RetrievalDecision(BaseModel):
    """检索闸门的判定结果。"""

    needs_retrieval: bool = Field(description="是否需要查询本地知识库来回答用户最新问题。")
    reason: str = Field(description="简短的中文判断理由。")


def knowledge_base_is_empty() -> bool:
    """知识库是否没有可检索的内容（无 child chunk）。"""
    try:
        status = knowledge_status()
    except Exception:
        return False
    return int(status.get("child_chunks", 0) or 0) <= 0


# 供 deep_research 等子图使用的 HybridRetriever 单例。
# 与 Self-RAG 图共享同一套 ChromaDB + BM25 索引，但独立于图的生成流程。
_RAG_RETRIEVER = None
_RAG_RETRIEVER_LOCK = threading.Lock()


def get_rag_retriever() -> HybridRetriever:
    """返回缓存的 HybridRetriever，用于纯检索场景（不做生成/评分/重写）。"""
    global _RAG_RETRIEVER
    if _RAG_RETRIEVER is None:
        with _RAG_RETRIEVER_LOCK:
            if _RAG_RETRIEVER is None:
                config = get_self_rag_config()
                config.use_hyde = False
                _RAG_RETRIEVER = HybridRetriever(config=config, llm=build_chat_backend(config))
    return _RAG_RETRIEVER


def decide_retrieval(messages: list[BaseMessage]) -> tuple[bool, dict[str, Any]]:
    """检索闸门：判断本轮是否需要进入实际的 RAG 检索流程。

    分两档：① 知识库为空直接短路（零 LLM）；② 否则用一次廉价的 LLM 分类，
    结合最近对话与领域描述判断。任何异常都保守地返回"需要检索"。
    返回 (should_retrieve, gate_trace)，gate_trace 用于可观测性。
    """
    if knowledge_base_is_empty():
        return False, {"decision": "direct", "tier": "empty_kb", "reason": "本地知识库为空，直接进行对话。"}

    recent_context = get_buffer_string(messages[-GATE_CONTEXT_TURNS:]) if messages else ""
    prompt = build_retrieval_gate_prompt(knowledge_domain=KNOWLEDGE_DOMAIN, recent_context=recent_context)
    try:
        decision = invoke_structured_json_schema(
            get_llm(),
            RetrievalDecision,
            [HumanMessage(content=prompt)],
            fallback_fn=lambda _: RetrievalDecision(needs_retrieval=True, reason="闸门判定失败，保守进行检索。"),
        )
    except Exception as exc:
        return True, {"decision": "retrieve", "tier": "error", "reason": f"闸门异常，保守检索：{exc}"}

    label = "retrieve" if decision.needs_retrieval else "direct"
    return decision.needs_retrieval, {"decision": label, "tier": "llm", "reason": decision.reason}


def get_self_rag_config() -> SelfRagConfig:
    """Return the OCT-agent-local Self-RAG config."""

    config = SelfRagConfig()
    config.chroma_dir = str(_resolve_data_path(config.chroma_dir, "chroma_store"))
    config.bm25_dir = str(_resolve_data_path(config.bm25_dir, "bm25_store"))
    config.sqlite_path = str(_resolve_data_path(config.sqlite_path, "rag_store", "rag.sqlite3"))
    config.parsed_dir = str(_resolve_data_path(config.parsed_dir, "rag_store", "parsed"))
    config.assets_dir = str(_resolve_data_path(config.assets_dir, "rag_store", "assets"))
    config.bm25_domain_terms_path = str(_resolve_data_path(config.bm25_domain_terms_path, "config", "domain_terms.txt"))
    if not config.rerank_model:
        config.use_rerank = False
    return config


def run_knowledge_query(question: str) -> dict[str, Any]:
    graph = _get_self_rag_graph()
    result = graph.invoke(initial_state(question))
    result["_used_chat_fallback"] = should_fallback_to_chat(result)
    return result


def self_rag_message(result: dict[str, Any]) -> AIMessage:
    generation = str(result.get("generation") or "").strip()
    error = str(result.get("error") or "").strip()
    if error and not generation:
        return AIMessage(content=f"本地知识库检索失败：{error}")

    citations = result.get("citations") or []
    content = generation or "本地知识库没有生成可用答案。"
    if citations:
        content += "\n\n## 本地引用\n"
        for citation in citations:
            label = citation.get("citation_id", "")
            title = citation.get("title") or Path(str(citation.get("source_path", ""))).name or "未知来源"
            section = citation.get("section", "")
            page = citation.get("page", "")
            location = "，".join(part for part in [section, f"页码 {page}" if page else ""] if part)
            content += f"- [{label}] {title}" + (f"（{location}）" if location else "") + "\n"
    return AIMessage(content=content.strip())


def should_fallback_to_chat(result: dict[str, Any]) -> bool:
    if result.get("error"):
        return True
    if not result.get("documents"):
        return True
    generation = str(result.get("generation") or "").lower()
    fallback_markers = [
        "cannot be determined from the provided context",
        "provided context does not contain",
        "无法从提供的上下文",
        "知识库没有",
        "无法判断",
    ]
    return any(marker in generation for marker in fallback_markers)


def ingest_knowledge_file(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_KNOWLEDGE_EXTENSIONS:
        raise ValueError(f"Unsupported knowledge file type: {path.suffix}")
    return ingest_file(str(path), config=get_self_rag_config())


def knowledge_status() -> dict[str, Any]:
    config = get_self_rag_config()
    sqlite_path = Path(config.sqlite_path)
    documents = 0
    parent_chunks = 0
    child_chunks = 0
    assets = 0
    references = 0
    if sqlite_path.exists():
        with sqlite3.connect(sqlite_path) as conn:
            documents = _count_table(conn, "documents")
            parent_chunks = _count_table(conn, "parent_chunks")
            child_chunks = _count_table(conn, "child_chunks")
            assets = _count_table(conn, "assets")
            references = _count_table(conn, '"references"')
    return {
        "data_dir": str(SELF_RAG_DATA_DIR),
        "sqlite_path": config.sqlite_path,
        "chroma_dir": config.chroma_dir,
        "bm25_dir": config.bm25_dir,
        "collection_name": config.collection_name,
        "uploads_dir": str(SELF_RAG_UPLOAD_DIR),
        "documents": documents,
        "parent_chunks": parent_chunks,
        "child_chunks": child_chunks,
        "assets": assets,
        "references": references,
        "initial_state_keys": sorted(initial_state("status").keys()),
    }


def _resolve_data_path(value: str, *default_parts: str) -> Path:
    path = Path(value) if value else SELF_RAG_DATA_DIR.joinpath(*default_parts)
    if path.is_absolute():
        return path
    return SELF_RAG_DATA_DIR / path


def _count_table(conn: sqlite3.Connection, table: str) -> int:
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.Error:
        return 0
