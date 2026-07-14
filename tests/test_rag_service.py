from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from embedding_client import EmbeddingClient
from rag_service import ask_from_vector_index
from vector_store import save_vector_index


def create_test_index(
    index_path: Path,
    dimension: int = 32,
) -> None:
    client = EmbeddingClient(
        model_name="test-embedding",
        dimension=dimension,
    )

    code_content = "def main(): print('hello')"
    doc_content = "This README explains how to run the project."

    records = [
        {
            "chunk_id": "main.py::chunk_0",
            "source_path": "main.py",
            "source_name": "main.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": 0,
            "content": code_content,
            "length": len(code_content),
            "embedding": client.embed_text(code_content),
        },
        {
            "chunk_id": "README.md::chunk_0",
            "source_path": "README.md",
            "source_name": "README.md",
            "source_suffix": ".md",
            "chunk_type": "document",
            "chunk_index": 0,
            "content": doc_content,
            "length": len(doc_content),
            "embedding": client.embed_text(doc_content),
        },
    ]

    save_vector_index(
        records=records,
        output_path=str(index_path),
    )


def test_ask_from_vector_index_success(tmp_path):
    index_path = tmp_path / "vector_index.json"
    create_test_index(index_path)

    result = ask_from_vector_index(
        query="def main(): print('hello')",
        index_path=str(index_path),
        top_k=1,
        embedding_model="test-embedding",
        dimension=32,
        chat_model="test-chat",
    )

    assert result["query"] == "def main(): print('hello')"
    assert result["chat_provider"] == "mock"
    assert result["chat_model"] == "test-chat"
    assert result["embedding_model"] == "test-embedding"
    assert result["top_k"] == 1
    assert result["retrieval_count"] == 1
    assert len(result["citations"]) == 1
    assert result["citations"][0]["citation_id"] == "Source 1"
    assert "[Source 1]" in result["answer"]
    assert "test-chat" in result["answer"]
    assert result["answer_quality"]["is_valid"] is True
    assert result["answer_quality"]["has_citations"] is True


def test_ask_from_vector_index_chunk_type_filter(tmp_path):
    index_path = tmp_path / "vector_index.json"
    create_test_index(index_path)

    result = ask_from_vector_index(
        query="README project",
        index_path=str(index_path),
        top_k=5,
        embedding_model="test-embedding",
        dimension=32,
        chunk_type="document",
    )

    assert result["chunk_type"] == "document"
    assert result["retrieval_count"] == 1
    assert result["citations"][0]["chunk_type"] == "document"


def test_ask_from_vector_index_index_not_exists():
    with pytest.raises(FileNotFoundError):
        ask_from_vector_index(
            query="main",
            index_path="not_exists_vector_index.json",
        )


def test_ask_from_vector_index_empty_query(tmp_path):
    index_path = tmp_path / "vector_index.json"
    create_test_index(index_path)

    with pytest.raises(ValueError):
        ask_from_vector_index(
            query="   ",
            index_path=str(index_path),
            dimension=32,
        )


def test_ask_from_vector_index_invalid_context_limit(tmp_path):
    index_path = tmp_path / "vector_index.json"
    create_test_index(index_path)

    with pytest.raises(ValueError):
        ask_from_vector_index(
            query="main",
            index_path=str(index_path),
            dimension=32,
            max_context_chars=0,
        )
