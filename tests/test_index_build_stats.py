from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from chunk_storage import save_chunks_to_json
from embedding_client import EmbeddingClient
from index_service import build_vector_index_from_json, build_vector_records
from vector_store import load_vector_index_bundle, save_vector_index


def build_chunks(count: int):
    return [
        {
            "chunk_id": f"file.py::chunk_{index}",
            "source_path": "file.py",
            "source_name": "file.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": index,
            "content": f"def func_{index}(): pass",
            "length": 20,
        }
        for index in range(count)
    ]


def test_build_vector_records_returns_batch_stats():
    client = EmbeddingClient(dimension=16)

    records, stats = build_vector_records(
        chunks=build_chunks(5),
        embedding_client=client,
        batch_size=2,
    )

    assert len(records) == 5
    assert stats["chunk_count"] == 5
    assert stats["vector_count"] == 5
    assert stats["batch_size"] == 2
    assert stats["batch_count"] == 3
    assert stats["request_count"] == 0
    assert stats["retry_count"] == 0
    assert stats["duration_ms"] >= 0


def test_build_vector_index_metadata_contains_build_stats(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    output_path = tmp_path / "vector_index.json"

    save_chunks_to_json(
        chunks=build_chunks(5),
        output_path=str(chunks_path),
    )

    result = build_vector_index_from_json(
        chunks_path=str(chunks_path),
        output_path=str(output_path),
        dimension=16,
        batch_size=2,
    )

    bundle = load_vector_index_bundle(str(output_path))

    assert result["build_stats"]["batch_count"] == 3
    assert bundle["metadata"]["build_stats"]["batch_size"] == 2
    assert bundle["metadata"]["build_stats"]["vector_count"] == 5


def test_save_vector_index_replaces_temp_file(tmp_path):
    output_path = tmp_path / "vector_index.json"
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    save_vector_index(
        records=[
            {
                "chunk_id": "a",
                "embedding": [1.0],
            }
        ],
        output_path=str(output_path),
        metadata={
            "index_format_version": "1.0",
            "embedding_provider": "mock",
            "embedding_model": "mock-hash-embedding",
            "dimension": 1,
        },
    )

    assert output_path.exists()
    assert not temp_path.exists()


def test_build_vector_index_reuses_unchanged_records(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    output_path = tmp_path / "vector_index.json"

    save_chunks_to_json(
        chunks=build_chunks(3),
        output_path=str(chunks_path),
    )

    first_result = build_vector_index_from_json(
        chunks_path=str(chunks_path),
        output_path=str(output_path),
        dimension=16,
        batch_size=2,
    )
    second_result = build_vector_index_from_json(
        chunks_path=str(chunks_path),
        output_path=str(output_path),
        dimension=16,
        batch_size=2,
    )

    assert first_result["update_stats"]["new_count"] == 3
    assert second_result["update_stats"]["new_count"] == 0
    assert second_result["update_stats"]["updated_count"] == 0
    assert second_result["update_stats"]["reused_count"] == 3
    assert second_result["update_stats"]["unique_embedding_count"] == 0


def test_build_vector_index_can_disable_incremental(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    output_path = tmp_path / "vector_index.json"

    save_chunks_to_json(
        chunks=build_chunks(2),
        output_path=str(chunks_path),
    )

    build_vector_index_from_json(
        chunks_path=str(chunks_path),
        output_path=str(output_path),
        dimension=16,
        batch_size=2,
    )
    result = build_vector_index_from_json(
        chunks_path=str(chunks_path),
        output_path=str(output_path),
        dimension=16,
        batch_size=2,
        incremental=False,
    )

    assert result["incremental"] is False
    assert result["update_stats"]["old_record_count"] == 0
    assert result["update_stats"]["new_count"] == 2
