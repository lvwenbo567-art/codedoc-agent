from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))


import pipelines.retrieval_pipeline as retrieval_pipeline


class FakeRewriteService:
    def rewrite(self, query: str, rewrite_count: int) -> dict:
        return {
            "original_query": query,
            "rewritten_queries": ["rewrite one", "rewrite two"],
            "rewritten_query": "rewrite one",
            "rewrite_applied": True,
            "protected_terms": [],
            "fallback_used": False,
            "fallback_reason": None,
        }


class DuplicateRewriteService:
    def rewrite(self, query: str, rewrite_count: int) -> dict:
        return {
            "original_query": query,
            "rewritten_queries": [query, "rewrite one", "rewrite one", "  "],
            "rewritten_query": query,
            "rewrite_applied": True,
            "protected_terms": [],
            "fallback_used": False,
            "fallback_reason": None,
        }


class CountingRerankClient:
    def __init__(self):
        self.call_count = 0

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.call_count += 1
        return [float(index) for index in range(len(documents))]


def test_multi_query_only_reranks_once(monkeypatch):
    def fake_hybrid_search(query: str, final_top_k: int, **kwargs) -> dict:
        return {
            "results": [
                {
                    "chunk_id": f"{query}-chunk",
                    "content": query,
                    "rank": 1,
                    "final_score": 1.0,
                }
            ]
        }

    monkeypatch.setattr(
        retrieval_pipeline,
        "hybrid_search_from_files",
        fake_hybrid_search,
    )

    rerank_client = CountingRerankClient()

    result = retrieval_pipeline.retrieve_with_rerank(
        query="original query",
        chunks_path="chunks.json",
        index_path="index.json",
        candidate_top_k=10,
        final_top_k=3,
        query_strategy="multi_query",
        rewrite_count=2,
        query_rewrite_service=FakeRewriteService(),
        rerank_client_override=rerank_client,
    )

    assert rerank_client.call_count == 1
    assert result["query_strategy"] == "multi_query"
    assert len(result["query_items"]) == 3
    assert result["candidate_count"] == 3


def test_rewrite_strategy_uses_rewritten_query(monkeypatch):
    called_queries = []

    def fake_hybrid_search(query: str, final_top_k: int, **kwargs) -> dict:
        called_queries.append(query)
        return {
            "dimension": 64,
            "results": [
                {
                    "chunk_id": "a",
                    "content": "A",
                    "rank": 1,
                    "final_score": 1.0,
                }
            ],
        }

    monkeypatch.setattr(
        retrieval_pipeline,
        "hybrid_search_from_files",
        fake_hybrid_search,
    )

    result = retrieval_pipeline.retrieve_with_rerank(
        query="original query",
        chunks_path="chunks.json",
        index_path="index.json",
        candidate_top_k=5,
        final_top_k=1,
        query_strategy="rewrite",
        query_rewrite_service=FakeRewriteService(),
        rerank_client_override=CountingRerankClient(),
    )

    assert called_queries == ["rewrite one"]
    assert result["query_items"][0]["query_type"] == "rewrite"


def test_multi_query_deduplicates_query_items(monkeypatch):
    def fake_hybrid_search(query: str, final_top_k: int, **kwargs) -> dict:
        return {
            "results": [
                {
                    "chunk_id": f"{query}-chunk",
                    "content": query,
                    "rank": 1,
                    "final_score": 1.0,
                }
            ]
        }

    monkeypatch.setattr(
        retrieval_pipeline,
        "hybrid_search_from_files",
        fake_hybrid_search,
    )

    result = retrieval_pipeline.retrieve_with_rerank(
        query="original query",
        chunks_path="chunks.json",
        index_path="index.json",
        candidate_top_k=10,
        final_top_k=2,
        query_strategy="multi_query",
        query_rewrite_service=DuplicateRewriteService(),
        rerank_client_override=CountingRerankClient(),
    )

    assert result["query_items"] == [
        {"query": "original query", "query_type": "original"},
        {"query": "rewrite one", "query_type": "rewrite"},
    ]
