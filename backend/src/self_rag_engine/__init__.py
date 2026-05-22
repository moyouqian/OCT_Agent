"""Independent LangGraph Self-RAG project."""

from .config import SelfRagConfig

__all__ = ["SelfRagConfig", "build_self_rag_graph", "run_self_rag"]


def __getattr__(name):
    if name in {"build_self_rag_graph", "run_self_rag"}:
        from .graph import build_self_rag_graph, run_self_rag

        return {
            "build_self_rag_graph": build_self_rag_graph,
            "run_self_rag": run_self_rag,
        }[name]
    raise AttributeError(name)
