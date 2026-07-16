from pathlib import Path
import math
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from chunk_storage import save_chunks_to_json
from embedding_client import EmbeddingClient
from index_service import build_vector_index_from_json, build_vector_records
from vector_store import load_vector_index


def build_test_chunks():
    return [
        {
            "chunk_id": "main.py::chunk_0",
            "source_path": "main.py",
            "source_name": "main.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": 0,
            "content": "def main(): pass",
            "length": 16,
        },
        {
            "chunk_id": "README.md::chunk_0",
            "source_path": "README.md",
            "source_name": "README.md",
            "source_suffix": ".md",
            "chunk_type": "document",
            "chunk_index": 0,
            "content": "This is a test project.",
            "length": 23,
        },
    ]


def test_build_vector_records_preserves_metadata():
    chunks = build_test_chunks()
    client = EmbeddingClient(dimension=32)

    records, stats = build_vector_records(
        chunks=chunks,
        embedding_client=client,
    )

    assert len(records) == 2
    assert stats["chunk_count"] == 2
    assert stats["vector_count"] == 2
    assert stats["batch_count"] == 1
    assert records[0]["chunk_id"] == "main.py::chunk_0"
    assert records[0]["source_path"] == "main.py"
    assert records[0]["source_name"] == "main.py"
    assert records[0]["source_suffix"] == ".py"
    assert records[0]["chunk_type"] == "code"
    assert records[0]["chunk_index"] == 0
    assert records[0]["content"] == "def main(): pass"
    assert records[0]["length"] == 16
    assert len(records[0]["embedding"]) == 32


def test_build_vector_records_normalizes_embeddings():
    chunks = build_test_chunks()
    client = EmbeddingClient(dimension=32)

    records, stats = build_vector_records(
        chunks=chunks,
        embedding_client=client,
    )

    norm = math.sqrt(sum(value * value for value in records[0]["embedding"]))

    assert stats["duration_ms"] >= 0
    assert abs(norm - 1.0) < 1e-6


def test_build_vector_index_from_json(tmp_path):
    chunks = build_test_chunks()

    chunks_path = tmp_path / "chunks.json"
    output_path = tmp_path / "vector_index.json"

    save_chunks_to_json(
        chunks=chunks,
        output_path=str(chunks_path),
    )

    result = build_vector_index_from_json(
        chunks_path=str(chunks_path),
        output_path=str(output_path),
        model_name="test-model",
        dimension=32,
    )

    assert result["chunks_path"] == str(chunks_path)
    assert result["output_path"] == str(output_path)
    assert result["model_name"] == "test-model"
    assert result["dimension"] == 32
    assert result["chunk_count"] == 2
    assert result["vector_count"] == 2
    assert result["build_stats"]["chunk_count"] == 2
    assert result["build_stats"]["vector_count"] == 2
    assert result["index_metadata"]["build_stats"]["batch_count"] == 1

    records = load_vector_index(str(output_path))

    assert len(records) == 2
    assert len(records[0]["embedding"]) == 32
    assert records[0]["chunk_id"] == "main.py::chunk_0"
