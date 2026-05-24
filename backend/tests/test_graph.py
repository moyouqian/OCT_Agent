import json

from self_rag_engine.config import SelfRagConfig
from self_rag_engine.graph import build_self_rag_graph, initial_state


class FakeLLM:
    def generate(self, messages, temperature=0.0, max_tokens=1024, json_mode=False):
        content = "\n".join(message["content"] for message in messages)
        if "Rewrite this question" in content:
            return json.dumps({"rewritten_question": "rewritten enterprise rag question"})
        if "whether a retrieved document is relevant" in content:
            return json.dumps({"binary_score": "yes", "score": 1, "reason": "relevant"})
        if "fully supported" in content:
            return json.dumps({"binary_score": "yes", "score": 1, "reason": "grounded"})
        if "addresses the user's question" in content:
            return json.dumps({"binary_score": "yes", "score": 1, "reason": "useful"})
        if "short factual passage" in content:
            return "hypothetical enterprise rag passage"
        return "Based on the context: enterprise RAG uses hybrid retrieval."


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return {
            "documents": [
                {
                    "text": "enterprise RAG uses dense retrieval, BM25, RRF, and reranking.",
                    "meta": {"source": "test.md", "chunk_index": 0},
                    "score": 1.0,
                    "retrieval_method": "fake",
                }
            ],
            "trace": {"query": query, "final_results": ["doc"]},
        }


class EmptyThenDocRetriever:
    def __init__(self):
        self.calls = 0

    def retrieve(self, query):
        self.calls += 1
        if self.calls == 1:
            return {"documents": [], "trace": {"query": query, "final_results": []}}
        return {
            "documents": [
                {
                    "text": "rewritten query found a document.",
                    "meta": {"source": "test.md", "chunk_index": 1},
                    "score": 0.9,
                }
            ],
            "trace": {"query": query, "final_results": ["doc"]},
        }


def test_graph_generates_grounded_answer():
    config = SelfRagConfig(use_hyde=False, use_rerank=False)
    graph = build_self_rag_graph(config=config, chat_backend=FakeLLM(), retriever=FakeRetriever())
    result = graph.invoke(initial_state("What does enterprise RAG use?"))

    assert "hybrid retrieval" in result["generation"]
    assert result["documents"]
    assert result["retrieval_trace"]
    assert result["grading_trace"]
    assert result.get("error") is None


def test_graph_rewrites_query_when_no_documents():
    config = SelfRagConfig(use_hyde=False, use_rerank=False, max_retrieval_attempts=2)
    retriever = EmptyThenDocRetriever()
    graph = build_self_rag_graph(config=config, chat_backend=FakeLLM(), retriever=retriever)
    result = graph.invoke(initial_state("missing question"))

    assert retriever.calls == 2
    assert result["rewritten_question"] == "rewritten enterprise rag question"
    assert result["documents"]
    assert result.get("error") is None
