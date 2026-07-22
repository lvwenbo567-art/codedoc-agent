from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from services.chunk_storage import save_chunks_to_json
from services.index_service import build_vector_index_from_json
from pipelines.retrieval_pipeline import retrieve_with_rerank


def make_chunk(chunk_id: str, content: str, chunk_type: str = "code") -> dict:
    """
    构造 pipeline 测试用 chunk。
    """
    return {
        "chunk_id": chunk_id,
        "source_path": f"test_project/{chunk_id}.py",
        "source_name": f"{chunk_id}.py",
        "source_suffix": ".py",
        "chunk_type": chunk_type,
        "chunk_index": 0,
        "content": content,
        "length": len(content),
    }


def prepare_chunks_and_index(tmp_path) -> tuple[Path, Path]:
    """
    准备临时 chunks 和 mock 向量索引。
    """
    chunks_path = tmp_path / "chunks.json"
    index_path = tmp_path / "vector_index.json"
    chunks = [
        make_chunk(
            "database::chunk_0",
            "数据库保存 projects files chunks 三张表。",
        ),
        make_chunk(
            "embedding::chunk_0",
            "EmbeddingClient 负责把文本转换成向量。",
        ),
        make_chunk(
            "api::chunk_0",
            "FastAPI 提供 /rerank_search 接口。",
        ),
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


def test_retrieve_with_rerank_success(tmp_path):
    chunks_path, index_path = prepare_chunks_and_index(tmp_path)

    result = retrieve_with_rerank(
        query="EmbeddingClient",
        chunks_path=str(chunks_path),
        index_path=str(index_path),
        candidate_top_k=3,
        final_top_k=2,
        embedding_provider="mock",
        embedding_model="test-model",
        mock_dimension=32,
        rerank_provider="mock",
        rerank_model="mock",
    )

    assert result["retrieval_mode"] == "hybrid_rerank"
    assert result["candidate_count"] == 3
    assert result["result_count"] == 2
    assert result["retrieval_duration_ms"] >= 0
    assert result["rerank_duration_ms"] >= 0
    assert result["total_duration_ms"] >= 0
    assert result["results"][0]["chunk_id"] == "embedding::chunk_0"
    assert result["results"][0]["rank"] == 1
    assert "retrieval_rank" in result["results"][0]
    assert "rerank_score" in result["results"][0]
    assert result["results"][0]["matched_queries"] == ["EmbeddingClient"]
    assert result["results"][0]["matched_query_types"] == ["original"]
    assert result["results"][0]["query_match_count"] == 1


def test_retrieve_with_rerank_rejects_invalid_candidate_top_k(tmp_path):
    chunks_path, index_path = prepare_chunks_and_index(tmp_path)

    with pytest.raises(ValueError):
        retrieve_with_rerank(
            query="EmbeddingClient",
            chunks_path=str(chunks_path),
            index_path=str(index_path),
            candidate_top_k=0,
        )


def test_retrieve_with_rerank_rejects_final_top_k_larger_than_candidates(tmp_path):
    chunks_path, index_path = prepare_chunks_and_index(tmp_path)

    with pytest.raises(ValueError):
        retrieve_with_rerank(
            query="EmbeddingClient",
            chunks_path=str(chunks_path),
            index_path=str(index_path),
            candidate_top_k=2,
            final_top_k=3,
        )


def test_retrieve_with_rerank_rejects_empty_query(tmp_path):
    chunks_path, index_path = prepare_chunks_and_index(tmp_path)

    with pytest.raises(ValueError, match="query"):
        retrieve_with_rerank(
            query="   ",
            chunks_path=str(chunks_path),
            index_path=str(index_path),
        )


def test_retrieve_with_rerank_rejects_invalid_rewrite_count(tmp_path):
    chunks_path, index_path = prepare_chunks_and_index(tmp_path)

    with pytest.raises(ValueError, match="rewrite_count"):
        retrieve_with_rerank(
            query="EmbeddingClient",
            chunks_path=str(chunks_path),
            index_path=str(index_path),
            rewrite_count=0,
        )
