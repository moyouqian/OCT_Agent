from typing import Any, Dict, List, Protocol


Document = Dict[str, Any]
Trace = Dict[str, Any]


class ChatBackend(Protocol):
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        ...


class Embedder(Protocol):
    def get_embedding(self, text: str) -> List[float]:
        ...


class Retriever(Protocol):
    def retrieve(self, query: str) -> Dict[str, Any]:
        ...
