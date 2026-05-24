from self_rag_engine.config import SelfRagConfig
from self_rag_engine.rerank import CloudReranker, build_reranker


def test_cloud_reranker_orders_results_by_provider_indexes():
    requests = []

    def fake_post(url, payload, headers):
        requests.append((url, payload, headers))
        return {
            "results": [
                {"index": 1, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.12},
            ]
        }

    reranker = CloudReranker(
        api_key="sk-test",
        model="Qwen/Qwen3-Reranker-4B",
        base_url="https://api.siliconflow.cn/v1",
        instruction="Rank passages.",
        http_post=fake_post,
    )
    passages = [
        {"text": "less relevant", "score": 0.5, "meta": {"child_id": "c1"}},
        {"text": "more relevant", "score": 0.4, "meta": {"child_id": "c2"}},
    ]

    results = reranker.rerank("query", passages, top_k=2)

    assert [item["text"] for item in results] == ["more relevant", "less relevant"]
    assert results[0]["rerank_score"] == 0.91
    assert results[0]["original_score"] == 0.4
    assert results[0]["rerank_model"] == "Qwen/Qwen3-Reranker-4B"
    assert requests[0][0] == "https://api.siliconflow.cn/v1/rerank"
    assert requests[0][1]["model"] == "Qwen/Qwen3-Reranker-4B"
    assert requests[0][1]["documents"] == ["less relevant", "more relevant"]
    assert requests[0][1]["top_n"] == 2
    assert requests[0][1]["return_documents"] is False
    assert requests[0][2]["Authorization"] == "Bearer sk-test"


def test_build_reranker_returns_none_when_disabled():
    config = SelfRagConfig(use_rerank=False)

    assert build_reranker(config) is None
