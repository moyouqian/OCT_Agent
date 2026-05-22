from pathlib import Path
from typing import Any, Dict, List, Optional
import pickle

from rank_bm25 import BM25Okapi

from .config import SelfRagConfig
from .llm import build_embedding_backend
from .memory_store import ChromaMemoryStore
from .parent_store import SQLiteParentStore
from .prompts import hyde_messages
from .rerank import build_reranker
from .text_utils import tokenize_for_search
from .types import ChatBackend, Document


class HybridRetriever:
    """Dense + BM25 + RRF + optional HyDE + optional rerank retriever."""

    def __init__(
        self,
        config: SelfRagConfig,
        llm: ChatBackend,
        embedder: Optional[Any] = None,
        memory: Optional[ChromaMemoryStore] = None,
        parent_store: Optional[SQLiteParentStore] = None,
        reranker: Optional[Any] = None,
    ):
        self.config = config
        self.llm = llm
        self.embedder = embedder or build_embedding_backend(config)
        self.memory = memory or ChromaMemoryStore(config.chroma_dir, config.collection_name)
        self.parent_store = parent_store or SQLiteParentStore(config.sqlite_path)
        self.reranker = reranker
        if self.reranker is None:
            self.reranker = build_reranker(config)

        self.bm25_index = None
        self.bm25_documents: List[str] = []
        self.bm25_metadata: List[Dict[str, Any]] = []
        self._load_or_build_bm25_index()

    def retrieve(self, query: str) -> Dict[str, Any]:
        standard = self.retrieve_once(query)
        result_lists = [standard["documents"]]
        trace = {
            "query": query,
            "standard": standard["trace"],
            "hyde_passage": "",
            "hyde": None,
            "hyde_error": None,
            "final_results": [],
        }

        if self.config.use_hyde:
            try:
                hyde_passage = self.llm.generate(hyde_messages(query), temperature=0.0, max_tokens=300)
            except Exception as exc:
                hyde_passage = ""
                trace["hyde_error"] = str(exc)

            trace["hyde_passage"] = hyde_passage
            if hyde_passage.strip():
                hyde = self.retrieve_once(hyde_passage)
                trace["hyde"] = hyde["trace"]
                result_lists.append(hyde["documents"])

        if len(result_lists) == 1:
            final_results = result_lists[0]
        else:
            final_results = self.reciprocal_rank_fusion(result_lists)[: self.config.final_k]

        trace["final_results"] = final_results
        return {"documents": final_results, "trace": trace}

    def retrieve_once(self, query: str) -> Dict[str, Any]:
        dense_results = self._dense_search(query, self.config.retrieval_top_k // 2)
        bm25_results = self._bm25_search(query, self.config.retrieval_top_k // 2)
        if dense_results and bm25_results:
            fused = self.reciprocal_rank_fusion([dense_results, bm25_results])
        elif dense_results:
            fused = dense_results
        elif bm25_results:
            fused = bm25_results
        else:
            fused = []

        candidates = fused[: self.config.retrieval_top_k]
        if self.reranker:
            candidates = self.reranker.rerank(query, candidates, top_k=len(candidates))

        parent_results = self._expand_to_parents(candidates) if self.config.expand_to_parent else []
        final_results = parent_results or candidates[: self.config.final_k]
        trace = {
            "query": query,
            "dense_results": dense_results,
            "bm25_results": bm25_results,
            "fused_results": fused,
            "reranked_results": candidates,
            "parent_results": parent_results,
            "final_k": self.config.context_parent_top_k if parent_results else self.config.final_k,
        }
        return {"documents": final_results, "trace": trace}

    def _dense_search(self, query: str, top_k: int) -> List[Document]:
        embedding = self.embedder.get_embedding(query)
        raw_results = self.memory.retrieve_chunks(embedding, top_k=top_k)
        results = []
        for chunk in raw_results:
            distance = chunk.get("score")
            score = 1.0 / (1.0 + distance) if isinstance(distance, (int, float)) else distance
            results.append(
                {
                    "text": chunk["text"],
                    "meta": {**chunk.get("metadata", {}), "child_id": chunk.get("id") or chunk.get("metadata", {}).get("child_id")},
                    "score": score,
                    "retrieval_method": "dense",
                }
            )
        return results

    def _bm25_search(self, query: str, top_k: int) -> List[Document]:
        if not self.bm25_index:
            return []
        scores = self.bm25_index.get_scores(tokenize_for_search_config(query, self.config))
        top_indices = scores.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {
                        "text": self.bm25_documents[idx],
                        "meta": self.bm25_metadata[idx],
                        "score": float(scores[idx]),
                        "retrieval_method": "bm25",
                    }
                )
        return results

    def reciprocal_rank_fusion(self, result_lists: List[List[Document]], k: int = 60) -> List[Document]:
        fused_scores: Dict[str, Dict[str, Any]] = {}
        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                meta = doc.get("meta", {})
                doc_id = (
                    meta.get("parent_id")
                    or meta.get("child_id")
                    or f"{meta.get('source', '')}:{meta.get('chunk_index', '')}:{doc['text']}"
                )
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {"doc": doc, "score": 0.0, "methods": []}
                fused_scores[doc_id]["score"] += 1.0 / (k + rank + 1)
                fused_scores[doc_id]["methods"].append(doc.get("retrieval_method", "unknown"))

        sorted_items = sorted(fused_scores.values(), key=lambda item: item["score"], reverse=True)
        final_results = []
        for item in sorted_items:
            doc = dict(item["doc"])
            doc["rrf_score"] = item["score"]
            doc["retrieval_methods"] = sorted(set(item["methods"]))
            doc["score"] = item["score"]
            final_results.append(doc)
        return final_results

    def rebuild_bm25_index(self) -> None:
        self.bm25_documents, self.bm25_metadata, self.bm25_index = build_bm25_artifacts(self.parent_store, self.config)
        if self.bm25_index is None:
            self.bm25_index = None
            self.bm25_documents = []
            self.bm25_metadata = []
            return
        self._save_bm25_index()

    def _expand_to_parents(self, child_candidates: List[Document]) -> List[Document]:
        parent_ids = [
            doc.get("meta", {}).get("parent_id")
            for doc in child_candidates
            if doc.get("meta", {}).get("parent_id")
        ]
        if not parent_ids:
            return []

        parent_rows = self.parent_store.get_parents(parent_ids)
        parent_by_id = {parent["parent_id"]: parent for parent in parent_rows}
        assets_by_parent = self.parent_store.get_assets_for_parents(parent_ids)
        aggregated: Dict[str, Dict[str, Any]] = {}

        for child in child_candidates:
            meta = child.get("meta", {})
            parent_id = meta.get("parent_id")
            if not parent_id or parent_id not in parent_by_id:
                continue
            item = aggregated.setdefault(
                parent_id,
                {
                    "score": child.get("score", 0),
                    "methods": [],
                    "matched_child_ids": [],
                },
            )
            score = child.get("score")
            if isinstance(score, (int, float)) and score > item["score"]:
                item["score"] = score
            item["methods"].append(child.get("retrieval_method", "unknown"))
            child_id = meta.get("child_id")
            if child_id:
                item["matched_child_ids"].append(child_id)

        documents = []
        for parent_id in dict.fromkeys(parent_ids):
            if parent_id not in aggregated or parent_id not in parent_by_id:
                continue
            parent = parent_by_id[parent_id]
            citation_id = f"S{len(documents) + 1}"
            related_assets = [
                compact_asset(asset)
                for asset in assets_by_parent.get(parent_id, [])
            ]
            meta = {
                "citation_id": citation_id,
                "parent_id": parent_id,
                "doc_id": parent["doc_id"],
                "title": parent.get("title", ""),
                "source": parent.get("source_path", ""),
                "source_path": parent.get("source_path", ""),
                "source_type": parent.get("source_type", ""),
                "file_type": parent.get("file_type", ""),
                "section_path": parent.get("section_path", ""),
                "page_start": parent.get("page_start"),
                "page_end": parent.get("page_end"),
                "parent_index": parent.get("parent_index"),
                "matched_child_ids": sorted(set(aggregated[parent_id]["matched_child_ids"])),
                "related_assets": related_assets if self.config.show_related_assets else [],
            }
            documents.append(
                {
                    "text": parent["text"],
                    "meta": meta,
                    "score": aggregated[parent_id]["score"],
                    "retrieval_method": "parent_expanded",
                    "retrieval_methods": sorted(set(aggregated[parent_id]["methods"])),
                }
            )
            if len(documents) >= self.config.context_parent_top_k:
                break
        return documents

    def _load_or_build_bm25_index(self) -> None:
        bm25_path, docs_path, meta_path = self._bm25_paths()
        if bm25_path.exists() and docs_path.exists() and meta_path.exists():
            with bm25_path.open("rb") as handle:
                self.bm25_index = pickle.load(handle)
            with docs_path.open("rb") as handle:
                self.bm25_documents = pickle.load(handle)
            with meta_path.open("rb") as handle:
                self.bm25_metadata = pickle.load(handle)
        else:
            self.rebuild_bm25_index()

    def _save_bm25_index(self) -> None:
        bm25_path, docs_path, meta_path = self._bm25_paths()
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with bm25_path.open("wb") as handle:
            pickle.dump(self.bm25_index, handle)
        with docs_path.open("wb") as handle:
            pickle.dump(self.bm25_documents, handle)
        with meta_path.open("wb") as handle:
            pickle.dump(self.bm25_metadata, handle)

    def _bm25_paths(self):
        base = Path(self.config.bm25_dir)
        return base / "bm25_index.pkl", base / "bm25_docs.pkl", base / "bm25_meta.pkl"


def build_bm25_artifacts(parent_store: SQLiteParentStore, config: Optional[SelfRagConfig] = None):
    records = parent_store.list_child_search_documents()
    if not records:
        return [], [], None
    documents = [record["text"] for record in records]
    metadata = [record["metadata"] for record in records]
    index = BM25Okapi([tokenize_for_search_config(doc, config) for doc in documents])
    return documents, metadata, index


def rebuild_bm25_files(config: SelfRagConfig, parent_store: SQLiteParentStore) -> None:
    documents, metadata, index = build_bm25_artifacts(parent_store, config)
    bm25_path = Path(config.bm25_dir)
    bm25_path.mkdir(parents=True, exist_ok=True)
    with (bm25_path / "bm25_index.pkl").open("wb") as handle:
        pickle.dump(index, handle)
    with (bm25_path / "bm25_docs.pkl").open("wb") as handle:
        pickle.dump(documents, handle)
    with (bm25_path / "bm25_meta.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)


def compact_asset(asset: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "asset_id": asset.get("asset_id"),
        "asset_type": asset.get("asset_type"),
        "label": asset.get("label"),
        "caption": asset.get("caption"),
        "page_start": asset.get("page_start"),
        "page_end": asset.get("page_end"),
    }


def tokenize_for_search_config(text: str, config: Optional[SelfRagConfig]) -> List[str]:
    if config is None:
        return tokenize_for_search(text)
    return tokenize_for_search(
        text,
        tokenizer=config.bm25_tokenizer,
        domain_terms_path=config.bm25_domain_terms_path,
        enable_cjk_bigrams=config.bm25_enable_cjk_bigrams,
    )
