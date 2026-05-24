from typing import Any, Callable, Dict, List, Optional
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import SelfRagConfig
from .types import Document


class CrossEncoderReranker:
    """Local cross-encoder reranker."""

    def __init__(self, model_name_or_path: str = "./models/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install the rerank extra to use CrossEncoderReranker.") from exc

        self.model = CrossEncoder(model_name_or_path, device=device)

    def rerank(self, query: str, passages: List[Document], top_k: int = 10) -> List[Document]:
        if not passages:
            return []

        pairs = [(query, p["text"]) for p in passages]
        scores = self.model.predict(pairs)
        scored = []
        for passage, score in zip(passages, scores):
            item = dict(passage)
            item["original_score"] = item.get("score", 0)
            item["rerank_score"] = float(score)
            item["score"] = float(score)
            scored.append(item)
        return sorted(scored, key=lambda p: p["score"], reverse=True)[:top_k]


class CloudReranker:
    """OpenAI-key-compatible reranker for SiliconFlow's /rerank endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        instruction: str = "",
        http_post: Optional[Callable[[str, Dict[str, Any], Dict[str, str]], Dict[str, Any]]] = None,
    ):
        if not api_key:
            raise RuntimeError("Cloud rerank requires SELF_RAG_OPENAI_API_KEY.")
        if not model:
            raise RuntimeError("Cloud rerank requires SELF_RAG_RERANK_MODEL.")
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.siliconflow.cn/v1").rstrip("/")
        self.instruction = instruction
        self.http_post = http_post or post_json

    def rerank(self, query: str, passages: List[Document], top_k: int = 10) -> List[Document]:
        if not passages:
            return []

        documents = [passage.get("text", "") for passage in passages]
        top_n = min(max(1, top_k), len(documents))
        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }
        if self.instruction:
            payload["instruction"] = self.instruction

        response = self.http_post(
            f"{self.base_url}/rerank",
            payload,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        results = response.get("results", [])
        scored = []
        for result in results:
            index = result.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(passages):
                continue
            score = float(result.get("relevance_score", result.get("score", 0.0)))
            item = dict(passages[index])
            item["original_score"] = item.get("score", 0)
            item["rerank_score"] = score
            item["rerank_model"] = self.model
            item["score"] = score
            scored.append(item)
        return scored[:top_n] if scored else passages[:top_n]


def build_reranker(config: SelfRagConfig):
    if not config.use_rerank:
        return None

    backend = config.rerank_backend.strip().lower()
    if backend == "cloud":
        return CloudReranker(
            api_key=config.openai_api_key,
            model=config.rerank_model,
            base_url=config.openai_base_url,
            instruction=config.rerank_instruction,
        )
    if backend == "local":
        return CrossEncoderReranker(model_name_or_path=config.rerank_model)
    raise ValueError("SELF_RAG_RERANK_BACKEND must be cloud or local.")


def post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Rerank request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Rerank request failed: {exc}") from exc
