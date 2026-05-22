"""Adapters for the built-in Self-RAG knowledge base."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from self_rag_engine.config import DEFAULT_SELF_RAG_DATA_DIR, SelfRagConfig
from self_rag_engine.graph import initial_state, run_self_rag
from self_rag_engine.ingestion import ingest_file


load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env")

SUPPORTED_KNOWLEDGE_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}
SELF_RAG_DATA_DIR = DEFAULT_SELF_RAG_DATA_DIR
SELF_RAG_UPLOAD_DIR = SELF_RAG_DATA_DIR / "uploads"


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
        config.use_cross_encoder = False
    return config


def run_knowledge_query(question: str) -> dict[str, Any]:
    config = get_self_rag_config()
    result = run_self_rag(question, config=config)
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
