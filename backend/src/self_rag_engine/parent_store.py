from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import sqlite3
from datetime import datetime, timezone


JSON_FIELDS = {"authors", "metadata", "bbox", "related_assets", "references"}


class SQLiteParentStore:
    """SQLite store for source documents, parent chunks, child chunks, and assets."""

    def __init__(self, sqlite_path: str = "./rag_store/rag.sqlite3"):
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL UNIQUE,
                    source_type TEXT,
                    file_type TEXT,
                    title TEXT,
                    authors_json TEXT,
                    year TEXT,
                    content_hash TEXT NOT NULL,
                    parser TEXT,
                    parser_version TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS parent_chunks (
                    parent_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    parent_index INTEGER NOT NULL,
                    title TEXT,
                    source_path TEXT,
                    source_type TEXT,
                    file_type TEXT,
                    section_path TEXT,
                    section_level INTEGER,
                    page_start INTEGER,
                    page_end INTEGER,
                    text TEXT NOT NULL,
                    metadata_json TEXT,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_parent_chunks_doc_id
                ON parent_chunks(doc_id);

                CREATE TABLE IF NOT EXISTS child_chunks (
                    child_id TEXT PRIMARY KEY,
                    parent_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    child_index INTEGER NOT NULL,
                    title TEXT,
                    source_path TEXT,
                    source_type TEXT,
                    file_type TEXT,
                    section_path TEXT,
                    page_start INTEGER,
                    page_end INTEGER,
                    text TEXT NOT NULL,
                    embedding_text TEXT NOT NULL,
                    metadata_json TEXT,
                    chroma_collection TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(parent_id) REFERENCES parent_chunks(parent_id) ON DELETE CASCADE,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_child_chunks_doc_id
                ON child_chunks(doc_id);

                CREATE INDEX IF NOT EXISTS idx_child_chunks_parent_id
                ON child_chunks(parent_id);

                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    doc_id TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    asset_index INTEGER,
                    label TEXT,
                    caption TEXT,
                    page_start INTEGER,
                    page_end INTEGER,
                    bbox_json TEXT,
                    file_path TEXT,
                    text_repr TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(parent_id) REFERENCES parent_chunks(parent_id) ON DELETE SET NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_assets_parent_id
                ON assets(parent_id);

                CREATE TABLE IF NOT EXISTS "references" (
                    reference_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    label TEXT,
                    raw_text TEXT,
                    title TEXT,
                    authors_json TEXT,
                    year TEXT,
                    doi TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_references_doc_id
                ON "references"(doc_id);
                """
            )

    def upsert_document(self, record: Dict[str, Any]) -> None:
        now = utc_now()
        payload = {
            "doc_id": record["doc_id"],
            "source_path": record["source_path"],
            "source_type": record.get("source_type", ""),
            "file_type": record.get("file_type", ""),
            "title": record.get("title", ""),
            "authors_json": json_dumps(record.get("authors", [])),
            "year": record.get("year", ""),
            "content_hash": record["content_hash"],
            "parser": record.get("parser", ""),
            "parser_version": record.get("parser_version", ""),
            "metadata_json": json_dumps(record.get("metadata", {})),
            "created_at": record.get("created_at") or now,
            "updated_at": now,
        }
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, source_path, source_type, file_type, title, authors_json,
                    year, content_hash, parser, parser_version, metadata_json,
                    created_at, updated_at
                )
                VALUES (
                    :doc_id, :source_path, :source_type, :file_type, :title, :authors_json,
                    :year, :content_hash, :parser, :parser_version, :metadata_json,
                    :created_at, :updated_at
                )
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_path=excluded.source_path,
                    source_type=excluded.source_type,
                    file_type=excluded.file_type,
                    title=excluded.title,
                    authors_json=excluded.authors_json,
                    year=excluded.year,
                    content_hash=excluded.content_hash,
                    parser=excluded.parser,
                    parser_version=excluded.parser_version,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                payload,
            )

    def upsert_parent(self, record: Dict[str, Any]) -> None:
        payload = {
            "parent_id": record["parent_id"],
            "doc_id": record["doc_id"],
            "parent_index": record.get("parent_index", 0),
            "title": record.get("title", ""),
            "source_path": record.get("source_path", ""),
            "source_type": record.get("source_type", ""),
            "file_type": record.get("file_type", ""),
            "section_path": record.get("section_path", ""),
            "section_level": record.get("section_level"),
            "page_start": record.get("page_start"),
            "page_end": record.get("page_end"),
            "text": record["text"],
            "metadata_json": json_dumps(record.get("metadata", {})),
            "content_hash": record["content_hash"],
            "created_at": record.get("created_at") or utc_now(),
        }
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO parent_chunks (
                    parent_id, doc_id, parent_index, title, source_path, source_type,
                    file_type, section_path, section_level, page_start, page_end,
                    text, metadata_json, content_hash, created_at
                )
                VALUES (
                    :parent_id, :doc_id, :parent_index, :title, :source_path, :source_type,
                    :file_type, :section_path, :section_level, :page_start, :page_end,
                    :text, :metadata_json, :content_hash, :created_at
                )
                ON CONFLICT(parent_id) DO UPDATE SET
                    parent_index=excluded.parent_index,
                    title=excluded.title,
                    source_path=excluded.source_path,
                    source_type=excluded.source_type,
                    file_type=excluded.file_type,
                    section_path=excluded.section_path,
                    section_level=excluded.section_level,
                    page_start=excluded.page_start,
                    page_end=excluded.page_end,
                    text=excluded.text,
                    metadata_json=excluded.metadata_json,
                    content_hash=excluded.content_hash
                """,
                payload,
            )

    def upsert_child(self, record: Dict[str, Any]) -> None:
        payload = {
            "child_id": record["child_id"],
            "parent_id": record["parent_id"],
            "doc_id": record["doc_id"],
            "child_index": record.get("child_index", 0),
            "title": record.get("title", ""),
            "source_path": record.get("source_path", ""),
            "source_type": record.get("source_type", ""),
            "file_type": record.get("file_type", ""),
            "section_path": record.get("section_path", ""),
            "page_start": record.get("page_start"),
            "page_end": record.get("page_end"),
            "text": record["text"],
            "embedding_text": record.get("embedding_text") or record["text"],
            "metadata_json": json_dumps(record.get("metadata", {})),
            "chroma_collection": record.get("chroma_collection", ""),
            "created_at": record.get("created_at") or utc_now(),
        }
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO child_chunks (
                    child_id, parent_id, doc_id, child_index, title, source_path,
                    source_type, file_type, section_path, page_start, page_end,
                    text, embedding_text, metadata_json, chroma_collection, created_at
                )
                VALUES (
                    :child_id, :parent_id, :doc_id, :child_index, :title, :source_path,
                    :source_type, :file_type, :section_path, :page_start, :page_end,
                    :text, :embedding_text, :metadata_json, :chroma_collection, :created_at
                )
                ON CONFLICT(child_id) DO UPDATE SET
                    child_index=excluded.child_index,
                    title=excluded.title,
                    source_path=excluded.source_path,
                    source_type=excluded.source_type,
                    file_type=excluded.file_type,
                    section_path=excluded.section_path,
                    page_start=excluded.page_start,
                    page_end=excluded.page_end,
                    text=excluded.text,
                    embedding_text=excluded.embedding_text,
                    metadata_json=excluded.metadata_json,
                    chroma_collection=excluded.chroma_collection
                """,
                payload,
            )

    def upsert_asset(self, record: Dict[str, Any]) -> None:
        payload = {
            "asset_id": record["asset_id"],
            "parent_id": record.get("parent_id"),
            "doc_id": record["doc_id"],
            "asset_type": record["asset_type"],
            "asset_index": record.get("asset_index", 0),
            "label": record.get("label", ""),
            "caption": record.get("caption", ""),
            "page_start": record.get("page_start"),
            "page_end": record.get("page_end"),
            "bbox_json": json_dumps(record.get("bbox", [])),
            "file_path": record.get("file_path", ""),
            "text_repr": record.get("text_repr", ""),
            "metadata_json": json_dumps(record.get("metadata", {})),
            "created_at": record.get("created_at") or utc_now(),
        }
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO assets (
                    asset_id, parent_id, doc_id, asset_type, asset_index, label,
                    caption, page_start, page_end, bbox_json, file_path,
                    text_repr, metadata_json, created_at
                )
                VALUES (
                    :asset_id, :parent_id, :doc_id, :asset_type, :asset_index, :label,
                    :caption, :page_start, :page_end, :bbox_json, :file_path,
                    :text_repr, :metadata_json, :created_at
                )
                ON CONFLICT(asset_id) DO UPDATE SET
                    parent_id=excluded.parent_id,
                    asset_type=excluded.asset_type,
                    asset_index=excluded.asset_index,
                    label=excluded.label,
                    caption=excluded.caption,
                    page_start=excluded.page_start,
                    page_end=excluded.page_end,
                    bbox_json=excluded.bbox_json,
                    file_path=excluded.file_path,
                    text_repr=excluded.text_repr,
                    metadata_json=excluded.metadata_json
                """,
                payload,
            )

    def upsert_reference(self, record: Dict[str, Any]) -> None:
        payload = {
            "reference_id": record["reference_id"],
            "doc_id": record["doc_id"],
            "label": record.get("label", ""),
            "raw_text": record.get("raw_text", ""),
            "title": record.get("title", ""),
            "authors_json": json_dumps(record.get("authors", [])),
            "year": record.get("year", ""),
            "doi": record.get("doi", ""),
            "metadata_json": json_dumps(record.get("metadata", {})),
            "created_at": record.get("created_at") or utc_now(),
        }
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO "references" (
                    reference_id, doc_id, label, raw_text, title, authors_json,
                    year, doi, metadata_json, created_at
                )
                VALUES (
                    :reference_id, :doc_id, :label, :raw_text, :title, :authors_json,
                    :year, :doi, :metadata_json, :created_at
                )
                ON CONFLICT(reference_id) DO UPDATE SET
                    doc_id=excluded.doc_id,
                    label=excluded.label,
                    raw_text=excluded.raw_text,
                    title=excluded.title,
                    authors_json=excluded.authors_json,
                    year=excluded.year,
                    doi=excluded.doi,
                    metadata_json=excluded.metadata_json
                """,
                payload,
            )

    def get_document_by_source_path(self, source_path: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE source_path = ?", (source_path,)).fetchone()
        return decode_row(row) if row else None

    def get_parent(self, parent_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM parent_chunks WHERE parent_id = ?", (parent_id,)).fetchone()
        return decode_row(row) if row else None

    def get_parents(self, parent_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = list(dict.fromkeys(parent_ids))
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM parent_chunks WHERE parent_id IN ({placeholders})",
                ids,
            ).fetchall()
        by_id = {row["parent_id"]: decode_row(row) for row in rows}
        return [by_id[parent_id] for parent_id in ids if parent_id in by_id]

    def get_assets_for_parent(self, parent_id: str) -> List[Dict[str, Any]]:
        return self.get_assets_for_parents([parent_id]).get(parent_id, [])

    def get_references_for_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM \"references\" WHERE doc_id = ? ORDER BY label, reference_id",
                (doc_id,),
            ).fetchall()
        return [decode_row(row) for row in rows]

    def get_assets_for_parents(self, parent_ids: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
        ids = list(dict.fromkeys(parent_ids))
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM assets WHERE parent_id IN ({placeholders}) ORDER BY asset_index",
                ids,
            ).fetchall()
        grouped: Dict[str, List[Dict[str, Any]]] = {parent_id: [] for parent_id in ids}
        for row in rows:
            asset = decode_row(row)
            grouped.setdefault(asset.get("parent_id", ""), []).append(asset)
        return grouped

    def list_child_search_documents(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT child_id, parent_id, doc_id, embedding_text, metadata_json
                FROM child_chunks
                ORDER BY doc_id, child_index
                """
            ).fetchall()
        records = []
        for row in rows:
            record = decode_row(row)
            metadata = record.get("metadata", {})
            metadata.update(
                {
                    "child_id": record["child_id"],
                    "parent_id": record["parent_id"],
                    "doc_id": record["doc_id"],
                }
            )
            records.append({"text": record["embedding_text"], "metadata": metadata})
        return records

    def list_child_ids_for_doc(self, doc_id: str) -> List[str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT child_id FROM child_chunks WHERE doc_id = ?", (doc_id,)).fetchall()
        return [row["child_id"] for row in rows]

    def delete_document(self, doc_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM \"references\" WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM assets WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM child_chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM parent_chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def reset(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                DELETE FROM "references";
                DELETE FROM assets;
                DELETE FROM child_chunks;
                DELETE FROM parent_chunks;
                DELETE FROM documents;
                """
            )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def json_loads(value: Optional[str]) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def decode_row(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    for key in list(data):
        if key.endswith("_json"):
            public_key = key[: -len("_json")]
            data[public_key] = json_loads(data.pop(key))
    return data
