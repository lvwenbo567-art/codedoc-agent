import json
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from services.chunk_storage import save_chunks_to_json
from services.index_service import build_vector_index_from_json
from services.vector_search_service import validate_index_compatibility
from repositories.vector_store import (
    build_index_metadata,
    load_vector_index,
    load_vector_index_bundle,
    save_vector_index,
)


def test_save_and_load_vector_index_bundle_with_metadata(tmp_path):
    output_path = tmp_path / "vector_index.json"
    records = [
        {
            "chunk_id": "README.md::chunk_0",
            "content": "hello",
            "embedding": [1.0, 0.0],
        }
    ]
    metadata = build_index_metadata(
        embedding_provider="mock",
        embedding_model="mock-hash-embedding",
        dimension=2,
        normalized=True,
        record_count=1,
    )

    save_vector_index(
        records=records,
        output_path=str(output_path),
        metadata=metadata,
    )

    bundle = load_vector_index_bundle(str(output_path))

    assert bundle["metadata"]["index_format_version"] == "1.0"
    assert bundle["metadata"]["embedding_provider"] == "mock"
    assert bundle["metadata"]["dimension"] == 2
    assert bundle["records"] == records
    assert load_vector_index(str(output_path)) == records


def test_load_legacy_vector_index_bundle(tmp_path):
    output_path = tmp_path / "legacy_vector_index.json"
    records = [{"chunk_id": "a", "embedding": [1.0]}]
    output_path.write_text(
        json.dumps(records),
        encoding="utf-8",
    )

    bundle = load_vector_index_bundle(str(output_path))

    assert bundle["metadata"] == {"index_format_version": "legacy"}
    assert bundle["records"] == records


def test_build_vector_index_from_json_writes_metadata(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    output_path = tmp_path / "vector_index.json"
    chunks = [
        {
            "chunk_id": "main.py::chunk_0",
            "source_path": "main.py",
            "source_name": "main.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": 0,
            "content": "def main(): pass",
            "length": 16,
        }
    ]

    save_chunks_to_json(chunks, str(chunks_path))

    result = build_vector_index_from_json(
        chunks_path=str(chunks_path),
        output_path=str(output_path),
        model_name="test-model",
        dimension=16,
    )

    bundle = load_vector_index_bundle(str(output_path))

    assert result["embedding_provider"] == "mock"
    assert result["embedding_model"] == "test-model"
    assert result["dimension"] == 16
    assert bundle["metadata"]["embedding_provider"] == "mock"
    assert bundle["metadata"]["embedding_model"] == "test-model"
    assert bundle["metadata"]["dimension"] == 16
    assert bundle["metadata"]["record_count"] == 1


def test_validate_index_compatibility_rejects_mismatched_provider():
    metadata = build_index_metadata(
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        dimension=768,
        normalized=True,
        record_count=1,
    )

    with pytest.raises(ValueError, match="Provider 不一致"):
        validate_index_compatibility(
            metadata=metadata,
            query_provider="openai_compatible",
            query_model="nomic-embed-text",
            query_dimension=768,
        )


def test_validate_index_compatibility_rejects_legacy_index():
    with pytest.raises(ValueError, match="旧格式"):
        validate_index_compatibility(
            metadata={"index_format_version": "legacy"},
            query_provider="ollama",
            query_model="nomic-embed-text",
            query_dimension=768,
        )
