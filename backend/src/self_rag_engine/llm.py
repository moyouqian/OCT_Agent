import json
import os
from typing import Any, Dict, List, Optional

from .config import SelfRagConfig


class OpenAICompatibleChat:
    """Provider-neutral OpenAI-compatible chat completion backend."""

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use cloud chat.") from exc

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content or ""


class OpenAICompatibleEmbedding:
    """Embedding adapter exposing get_embedding for retrieval components."""

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use cloud embeddings.") from exc

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class SentenceTransformerEmbedder:
    """Local embedding fallback."""

    def __init__(self, model_name_or_path: str = "./models/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the ingestion or rerank extra to use local embeddings.") from exc

        self.model = SentenceTransformer(model_name_or_path)

    def get_embedding(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [vec.tolist() for vec in self.model.encode(texts)]


class LocalLlamaChat:
    """Local llama.cpp fallback for chat generation."""

    def __init__(self, model_path: str, n_gpu_layers: int = 20, n_threads: int = 6, n_ctx: int = 8192):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("Install the local-llm extra to use llama.cpp fallback.") from exc

        os.environ.setdefault("GGML_CUDA_MMQ", "1")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_ctx=n_ctx,
            offload_kqv=True,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        prompt = messages_to_prompt(messages)
        if json_mode:
            prompt += "\nReturn only valid JSON."
        self.llm.reset()
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            stop=["<|user|>", "<|assistant|>", "<|/assistant|>", "<|system|>"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()


def build_chat_backend(config: SelfRagConfig):
    backend = config.chat_backend.lower()
    if backend not in {"auto", "cloud", "local"}:
        raise ValueError("chat_backend must be auto, cloud, or local.")

    if backend in {"auto", "cloud"} and config.cloud_chat_configured():
        return OpenAICompatibleChat(
            api_key=config.openai_api_key,
            model=config.chat_model,
            base_url=config.openai_base_url or None,
        )

    if backend == "cloud":
        raise RuntimeError("Cloud chat requires SELF_RAG_OPENAI_API_KEY and SELF_RAG_CHAT_MODEL.")

    if not os.path.exists(config.local_model_path):
        raise RuntimeError(
            "No cloud chat backend is configured and local model file was not found: "
            f"{config.local_model_path}"
        )
    return LocalLlamaChat(config.local_model_path)


def build_embedding_backend(config: SelfRagConfig):
    backend = config.embedding_backend.lower()
    if backend not in {"auto", "cloud", "local"}:
        raise ValueError("embedding_backend must be auto, cloud, or local.")

    if backend in {"auto", "cloud"} and config.cloud_embedding_configured():
        return OpenAICompatibleEmbedding(
            api_key=config.openai_api_key,
            model=config.embedding_model,
            base_url=config.openai_base_url or None,
        )

    if backend == "cloud":
        raise RuntimeError("Cloud embeddings require SELF_RAG_OPENAI_API_KEY and SELF_RAG_EMBEDDING_MODEL.")

    return SentenceTransformerEmbedder()


def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"<|{role}|>\n{content.strip()}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def parse_json_object(raw: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
    result = dict(fallback)
    result["raw"] = raw
    return result
