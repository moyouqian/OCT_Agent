from typing import Any, Dict, List, Optional
import hashlib


class ChromaMemoryStore:
    """Persistent ChromaDB document chunk store."""

    def __init__(self, persist_dir: str = "./chroma_store", collection_name: str = "documents"):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install the ingestion extra to use ChromaMemoryStore.") from exc

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def retrieve_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 8,
        filters: Optional[Dict[str, Any]] = None,
        include_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        query_args: Dict[str, Any] = {"query_embeddings": [query_embedding], "n_results": top_k}
        if filters:
            query_args["where"] = filters

        results = self.collection.query(**query_args)
        distances = results["distances"][0] if include_scores else [None] * len(results["documents"][0])
        retrieved = []
        ids = results.get("ids", [[]])[0]
        for chunk_id, doc, meta, score in zip(ids, results["documents"][0], results["metadatas"][0], distances):
            retrieved.append({"id": chunk_id, "text": doc, "metadata": meta, "score": score})
        return retrieved

    def store_document(
        self,
        chunk: str,
        embedding: List[float],
        file_path: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        chunk_id = hashlib.sha256(f"{file_path}:{chunk}".encode("utf-8")).hexdigest()
        meta = {"source": file_path, "file_type": file_type.lstrip(".")}
        if metadata:
            meta.update(metadata)
        self.collection.add(documents=[chunk], embeddings=[embedding], ids=[chunk_id], metadatas=[clean_metadata(meta)])

    def store_child_chunk(
        self,
        child_id: str,
        document: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        self.collection.upsert(
            documents=[document],
            embeddings=[embedding],
            ids=[child_id],
            metadatas=[clean_metadata(metadata)],
        )

    def delete_ids(self, ids: List[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    def delete_where(self, filters: Dict[str, Any]) -> None:
        self.collection.delete(where=filters)

    def all_documents(self) -> Dict[str, Any]:
        return self.collection.get()

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    clean = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean
