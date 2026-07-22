from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from clients.rerank_client import RerankClient, RerankConfig
from services.rerank_service import get_candidate_content, rerank_candidates


class WrongCountClient:
    """
    模拟返回分数数量错误的 RerankClient。
    """

    def score(self, query, documents):
        return [0.1]


def test_get_candidate_content_prefers_content():
    candidate = {
        "content": "完整内容",
        "content_preview": "预览内容",
    }

    assert get_candidate_content(candidate) == "完整内容"


def test_get_candidate_content_falls_back_to_preview():
    candidate = {
        "chunk_id": "a",
        "content_preview": "预览内容",
    }

    assert get_candidate_content(candidate) == "预览内容"


def test_get_candidate_content_rejects_missing_text():
    with pytest.raises(ValueError):
        get_candidate_content({"chunk_id": "missing"})


def test_rerank_changes_order_and_keeps_retrieval_rank():
    candidates = [
        {
            "chunk_id": "a",
            "rank": 1,
            "content": "数据库内容",
        },
        {
            "chunk_id": "b",
            "rank": 2,
            "content": "EmbeddingClient 生成向量",
        },
    ]
    client = RerankClient(
        config=RerankConfig(
            provider="mock",
            model_name_or_path="mock",
        )
    )

    results = rerank_candidates(
        query="EmbeddingClient",
        candidates=candidates,
        rerank_client=client,
        final_top_k=2,
    )

    assert results[0]["chunk_id"] == "b"
    assert results[0]["retrieval_rank"] == 2
    assert results[0]["rank"] == 1
    assert "rerank_score" in results[0]


def test_rerank_candidates_returns_empty_for_no_candidates():
    client = RerankClient()

    assert rerank_candidates("query", [], client) == []


def test_rerank_candidates_rejects_empty_query():
    client = RerankClient()

    with pytest.raises(ValueError):
        rerank_candidates(" ", [{"content": "document"}], client)


def test_rerank_candidates_rejects_invalid_top_k():
    client = RerankClient()

    with pytest.raises(ValueError):
        rerank_candidates("query", [{"content": "document"}], client, final_top_k=0)


def test_rerank_candidates_rejects_wrong_score_count():
    with pytest.raises(ValueError):
        rerank_candidates(
            query="query",
            candidates=[
                {"chunk_id": "a", "content": "one"},
                {"chunk_id": "b", "content": "two"},
            ],
            rerank_client=WrongCountClient(),
            final_top_k=2,
        )
