from pathlib import Path

from self_rag_engine.config import SelfRagConfig
from self_rag_engine.parent_store import SQLiteParentStore
from self_rag_engine.retrieval import HybridRetriever


class FakeLLM:
    def generate(self, *args, **kwargs):
        return ""


class FakeEmbedder:
    def get_embedding(self, text):
        return [1.0, 0.0]


class FakeMemory:
    def retrieve_chunks(self, embedding, top_k=8):
        return [
            {
                "id": "child_1",
                "text": "Title: Paper\nSection: Methods\nContent:\nparent child chunking",
                "metadata": {
                    "child_id": "child_1",
                    "parent_id": "parent_1",
                    "doc_id": "doc_1",
                    "title": "Paper",
                    "section_path": "Methods",
                },
                "score": 0.2,
            }
        ]


def test_retriever_expands_child_hits_to_parent_documents(tmp_path: Path):
    config = SelfRagConfig(
        sqlite_path=str(tmp_path / "rag.sqlite3"),
        bm25_dir=str(tmp_path / "bm25"),
        use_hyde=False,
        use_rerank=False,
        retrieval_top_k=4,
        context_parent_top_k=2,
    )
    parent_store = SQLiteParentStore(config.sqlite_path)
    parent_store.upsert_document(
        {
            "doc_id": "doc_1",
            "source_path": str(tmp_path / "paper.pdf"),
            "source_type": "paper",
            "file_type": "pdf",
            "title": "Paper",
            "content_hash": "hash",
            "parser": "docling",
        }
    )
    parent_store.upsert_parent(
        {
            "parent_id": "parent_1",
            "doc_id": "doc_1",
            "parent_index": 0,
            "title": "Paper",
            "source_path": str(tmp_path / "paper.pdf"),
            "source_type": "paper",
            "file_type": "pdf",
            "section_path": "Methods",
            "section_level": 2,
            "page_start": 3,
            "page_end": 4,
            "text": "The method uses parent child chunking.",
            "content_hash": "parent_hash",
        }
    )

    retriever = HybridRetriever(
        config=config,
        llm=FakeLLM(),
        embedder=FakeEmbedder(),
        memory=FakeMemory(),
        parent_store=parent_store,
    )
    result = retriever.retrieve_once("chunking")

    assert result["documents"][0]["text"] == "The method uses parent child chunking."
    assert result["documents"][0]["meta"]["citation_id"] == "S1"
    assert result["documents"][0]["meta"]["page_start"] == 3
    assert result["documents"][0]["retrieval_method"] == "parent_expanded"
