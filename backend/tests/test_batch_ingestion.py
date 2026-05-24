from pathlib import Path

from self_rag_engine.config import SelfRagConfig
from self_rag_engine.ingestion import (
    ChunkQualityStats,
    discover_ingest_candidates,
    format_batch_report,
    quality_has_blockers,
    run_batch_ingestion,
)


def test_discover_ingest_candidates_filters_and_deduplicates(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    first = dataset / "first.md"
    duplicate = dataset / "duplicate.md"
    unsupported = dataset / "ignore.png"
    excluded = dataset / "skip.md"
    first.write_text("# Title\n\nUseful chunking content.", encoding="utf-8")
    duplicate.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
    unsupported.write_text("not supported", encoding="utf-8")
    excluded.write_text("# Skip\n\nExcluded content.", encoding="utf-8")

    candidates = discover_ingest_candidates(
        dataset_dirs=[str(dataset)],
        exclude_globs=["skip.md"],
    )
    by_name = {candidate.path.name: candidate for candidate in candidates}

    duplicate_statuses = {by_name["first.md"].status, by_name["duplicate.md"].status}

    assert duplicate_statuses == {"pending", "duplicate"}
    assert by_name["ignore.png"].reason == "unsupported_extension"
    assert by_name["skip.md"].reason == "matched_exclude_rule"


def test_dry_run_batch_ingestion_reports_chunk_quality(tmp_path: Path):
    source = tmp_path / "paper.md"
    source.write_text(
        "# Paper Title\n\n## Abstract\n\nThis paper studies retrieval.\n\n## Methods\n\nWe use parent child chunking.",
        encoding="utf-8",
    )
    config = SelfRagConfig(
        sqlite_path=str(tmp_path / "rag.sqlite3"),
        bm25_dir=str(tmp_path / "bm25"),
        parsed_dir=str(tmp_path / "parsed"),
        child_chunk_size=20,
        child_chunk_overlap=2,
    )
    candidates = discover_ingest_candidates(files=[str(source)])

    summary = run_batch_ingestion(candidates, config=config, dry_run=True)
    report = format_batch_report(summary)

    assert summary["dry_run"] is True
    assert summary["processed_files"] == 1
    assert summary["total_parents"] >= 2
    assert summary["total_children"] >= 2
    assert summary["ingested_children"] == 0
    assert "Quality gates" in report
    assert not (tmp_path / "rag.sqlite3").exists()


def test_batch_rebuilds_bm25_once(tmp_path: Path, monkeypatch):
    import self_rag_engine.ingestion as ing
    import self_rag_engine.retrieval as retr

    class FakeEmbedder:
        def get_embedding(self, text):
            return [1.0, 2.0]

    class FakeStore:
        def __init__(self, *args, **kwargs):
            self.records = {}

        def store_child_chunk(self, child_id, document, embedding, metadata):
            self.records[child_id] = 1

        def delete_ids(self, ids):
            pass

    monkeypatch.setattr(ing, "build_embedding_backend", lambda config: FakeEmbedder())
    monkeypatch.setattr(ing, "ChromaMemoryStore", lambda *a, **k: FakeStore())

    real = retr.rebuild_bm25_files
    calls = {"n": 0}

    def counting(cfg, ps):
        calls["n"] += 1
        return real(cfg, ps)

    monkeypatch.setattr(retr, "rebuild_bm25_files", counting)

    a = tmp_path / "a.md"
    a.write_text("# A\n\nAlpha content here for chunking.", encoding="utf-8")
    b = tmp_path / "b.md"
    b.write_text("# B\n\nBeta content here for chunking.", encoding="utf-8")
    config = SelfRagConfig(sqlite_path=str(tmp_path / "r.sqlite3"), bm25_dir=str(tmp_path / "bm25"))
    candidates = discover_ingest_candidates(files=[str(a), str(b)])

    run_batch_ingestion(candidates, config=config)

    assert calls["n"] == 1


def test_quality_gate_flags_blocking_warnings():
    clean = ChunkQualityStats(parent_chunks=1, child_chunks=1)
    noisy = ChunkQualityStats(parent_chunks=1, child_chunks=1, mojibake_warnings=1)

    assert quality_has_blockers(clean) is False
    assert quality_has_blockers(noisy) is True


def test_bm25_files_loadable_after_rebuild(tmp_path: Path):
    import pickle

    from self_rag_engine.parent_store import SQLiteParentStore
    from self_rag_engine.retrieval import rebuild_bm25_files

    config = SelfRagConfig(sqlite_path=str(tmp_path / "r.sqlite3"), bm25_dir=str(tmp_path / "bm25"))
    ps = SQLiteParentStore(config.sqlite_path)
    rebuild_bm25_files(config, ps)  # empty store should still write safely
    for name in ("bm25_index.pkl", "bm25_docs.pkl", "bm25_meta.pkl"):
        with (tmp_path / "bm25" / name).open("rb") as fh:
            pickle.load(fh)
