from pathlib import Path

from self_rag_engine.config import SelfRagConfig
from self_rag_engine.chunking import build_parent_chunks
from self_rag_engine.document_parser import Block, ParsedDocument
from self_rag_engine.document_parser import parse_markdown_file
from self_rag_engine.ingestion import ingest_file
from self_rag_engine.parent_store import SQLiteParentStore


class FakeEmbedder:
    def get_embedding(self, text):
        return [float(len(text) % 10), 1.0]


class FakeMemoryStore:
    def __init__(self):
        self.records = {}
        self.deleted_ids = []

    def store_child_chunk(self, child_id, document, embedding, metadata):
        self.records[child_id] = {
            "document": document,
            "embedding": embedding,
            "metadata": metadata,
        }

    def delete_ids(self, ids):
        self.deleted_ids.extend(ids)
        for child_id in ids:
            self.records.pop(child_id, None)


class FailingEmbedder:
    def get_embedding(self, text):
        raise RuntimeError("embedding failed")


class BatchEmbedder:
    def __init__(self):
        self.batch_calls = 0

    def get_embeddings(self, texts):
        self.batch_calls += 1
        return [[float(len(t) % 10), 1.0] for t in texts]


def test_markdown_ingestion_writes_parent_child_store(tmp_path: Path):
    source = tmp_path / "paper.md"
    source.write_text(
        "# Paper Title\n\n## Abstract\n\nThis paper studies retrieval.\n\n## Methods\n\nWe use parent child chunking.",
        encoding="utf-8",
    )
    config = SelfRagConfig(
        sqlite_path=str(tmp_path / "rag.sqlite3"),
        bm25_dir=str(tmp_path / "bm25"),
        child_chunk_size=20,
        child_chunk_overlap=2,
    )
    parent_store = SQLiteParentStore(config.sqlite_path)
    memory = FakeMemoryStore()

    result = ingest_file(str(source), config=config, embedder=FakeEmbedder(), store=memory, parent_store=parent_store)

    assert result["skipped"] == 0
    assert result["total_parents"] >= 2
    assert result["ingested_children"] == len(memory.records)
    child_records = parent_store.list_child_search_documents()
    assert child_records
    assert child_records[0]["metadata"]["parent_id"]
    assert (tmp_path / "bm25" / "bm25_docs.pkl").exists()


def test_ingestion_skips_unchanged_source(tmp_path: Path):
    source = tmp_path / "note.md"
    source.write_text("# Title\n\nStable content.", encoding="utf-8")
    config = SelfRagConfig(sqlite_path=str(tmp_path / "rag.sqlite3"), bm25_dir=str(tmp_path / "bm25"))
    parent_store = SQLiteParentStore(config.sqlite_path)
    memory = FakeMemoryStore()

    ingest_file(str(source), config=config, embedder=FakeEmbedder(), store=memory, parent_store=parent_store)
    second = ingest_file(str(source), config=config, embedder=FakeEmbedder(), store=memory, parent_store=parent_store)

    assert second["skipped"] == 1


def test_ingestion_rolls_back_parent_store_when_embedding_fails(tmp_path: Path):
    source = tmp_path / "note.md"
    source.write_text("# Title\n\nStable content.", encoding="utf-8")
    config = SelfRagConfig(sqlite_path=str(tmp_path / "rag.sqlite3"), bm25_dir=str(tmp_path / "bm25"))
    parent_store = SQLiteParentStore(config.sqlite_path)
    memory = FakeMemoryStore()

    try:
        ingest_file(str(source), config=config, embedder=FailingEmbedder(), store=memory, parent_store=parent_store)
    except RuntimeError:
        pass

    assert parent_store.get_document_by_source_path(str(source.resolve())) is None
    assert parent_store.list_child_search_documents() == []
    assert memory.records == {}


def test_ingestion_uses_batch_embeddings(tmp_path: Path):
    source = tmp_path / "p.md"
    source.write_text("# T\n\n## A\n\nalpha content.\n\n## B\n\nbeta content.", encoding="utf-8")
    config = SelfRagConfig(
        sqlite_path=str(tmp_path / "r.sqlite3"),
        bm25_dir=str(tmp_path / "bm25"),
        child_chunk_size=20,
        child_chunk_overlap=2,
    )
    parent_store = SQLiteParentStore(config.sqlite_path)
    memory = FakeMemoryStore()
    embedder = BatchEmbedder()

    result = ingest_file(str(source), config=config, embedder=embedder, store=memory, parent_store=parent_store)

    assert embedder.batch_calls >= 1
    assert result["ingested_children"] == len(memory.records)


def test_ingestion_skips_duplicate_content_other_path(tmp_path: Path):
    body = "# Same\n\nIdentical body content."
    a = tmp_path / "a.md"
    a.write_text(body, encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    b = sub / "b.md"
    b.write_text(body, encoding="utf-8")
    config = SelfRagConfig(sqlite_path=str(tmp_path / "r.sqlite3"), bm25_dir=str(tmp_path / "bm25"))
    ps = SQLiteParentStore(config.sqlite_path)
    mem = FakeMemoryStore()

    ingest_file(str(a), config=config, embedder=FakeEmbedder(), store=mem, parent_store=ps)
    second = ingest_file(str(b), config=config, embedder=FakeEmbedder(), store=mem, parent_store=ps)

    assert second["skipped"] == 1
    assert second.get("duplicate_of")


def test_markdown_parser_preserves_heading_blocks(tmp_path: Path):
    source = tmp_path / "paper.md"
    source.write_text("# Title\n\n## Methods\n\nDetails.", encoding="utf-8")

    parsed = parse_markdown_file(source)

    assert parsed.title == "Title"
    assert [block.block_type for block in parsed.blocks[:2]] == ["title", "section_header"]


def test_parent_chunking_skips_reference_like_sections():
    parsed = ParsedDocument(
        doc_id="doc_test",
        source_path="paper.pdf",
        source_type="file",
        file_type="pdf",
        title="Paper",
        content_hash="hash",
        parser="test",
        blocks=[
            Block(block_id="b1", block_type="section_header", text="Methods", level=2),
            Block(block_id="b2", block_type="paragraph", text="Useful method text for retrieval."),
            Block(block_id="b3", block_type="section_header", text="参考文献：", level=2),
            Block(
                block_id="b4",
                block_type="paragraph",
                text="[1] HUANG D, SWANSON E A, LIN C P, et al. Optical coherence tomography [J]. Science, 1991.",
            ),
        ],
    )

    parents, _ = build_parent_chunks(parsed, SelfRagConfig(parent_chunk_max_tokens=200))

    assert [parent.section_path for parent in parents] == ["Methods"]
    assert all("HUANG D" not in parent.text for parent in parents)
