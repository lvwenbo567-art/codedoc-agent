from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from chunk_storage import save_chunks_to_json
from index_service import build_vector_index_from_json
from rerank_client import RerankServiceError
from retrieval_pipeline import retrieve_with_rerank


class BrokenReranker:
    """
    模拟 Rerank 服务不可用。
    """

    def score(self, query: str, documents: list[str]) -> list[float]:
        raise RerankServiceError("模拟模型不可用")


class WrongCountReranker:
    """
    模拟业务 bug：Rerank 返回分数数量不对。
    """

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [1.0]


def make_chunk(chunk_id: str, content: str) -> dict:
    """
    构造 fallback 测试用 chunk。
    """
    return {
        "chunk_id": chunk_id,
        "source_path": f"{chunk_id}.py",
        "source_name": f"{chunk_id}.py",
        "source_suffix": ".py",
        "chunk_type": "code",
        "chunk_index": 0,
        "content": content,
        "length": len(content),
    }


def prepare_index(tmp_path) -> tuple[Path, Path]:
    """
    准备临时 chunks 和 mock 向量索引。
    """
    chunks_path = tmp_path / "chunks.json"
    index_path = tmp_path / "vector_index.json"
    chunks = [
        make_chunk("embedding::chunk_0", "EmbeddingClient 负责生成向量。"),
        make_chunk("database::chunk_0", "SQLite 保存 projects files chunks。"),
    ]

    save_chunks_to_json(chunks, str(chunks_path))
    build_vector_index_from_json(
        chunks_path=str(chunks_path),
        output_path=str(index_path),
        embedding_provider="mock",
        embedding_model="test-model",
        mock_dimension=32,
        incremental=False,
    )

    return chunks_path, index_path


def test_retrieve_with_rerank_falls_back_to_hybrid_on_service_error(
    tmp_path,
    monkeypatch,
):
    chunks_path, index_path = prepare_index(tmp_path)

    monkeypatch.setattr(
        "retrieval_pipeline.get_cached_rerank_client",
        lambda config: BrokenReranker(),
    )

    result = retrieve_with_rerank(
        query="EmbeddingClient",
        chunks_path=str(chunks_path),
        index_path=str(index_path),
        candidate_top_k=2,
        final_top_k=1,
        embedding_provider="mock",
        embedding_model="test-model",
        mock_dimension=32,
    )

    assert result["degraded"] is True
    assert result["rerank_applied"] is False
    assert result["retrieval_mode"] == "hybrid_fallback"
    assert "模拟模型不可用" in result["degrade_reason"]
    assert result["rerank_duration_ms"] >= 0
    assert result["result_count"] == 1


def test_retrieve_with_rerank_does_not_hide_value_error(
    tmp_path,
    monkeypatch,
):
    chunks_path, index_path = prepare_index(tmp_path)

    monkeypatch.setattr(
        "retrieval_pipeline.get_cached_rerank_client",
        lambda config: WrongCountReranker(),
    )

    with pytest.raises(ValueError):
        retrieve_with_rerank(
            query="EmbeddingClient",
            chunks_path=str(chunks_path),
            index_path=str(index_path),
            candidate_top_k=2,
            final_top_k=2,
            embedding_provider="mock",
            embedding_model="test-model",
            mock_dimension=32,
        )


def test_retrieve_with_rerank_success_has_stability_fields(tmp_path):
    chunks_path, index_path = prepare_index(tmp_path)

    result = retrieve_with_rerank(
        query="EmbeddingClient",
        chunks_path=str(chunks_path),
        index_path=str(index_path),
        candidate_top_k=2,
        final_top_k=1,
        embedding_provider="mock",
        embedding_model="test-model",
        mock_dimension=32,
        rerank_provider="mock",
        rerank_model="mock",
    )

    assert result["retrieval_mode"] == "hybrid_rerank"
    assert result["rerank_applied"] is True
    assert result["degraded"] is False
    assert result["degrade_reason"] is None
    assert result["rerank_duration_ms"] >= 0
