from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from chunk_storage import (
    calculate_chunk_stats,
    load_chunks_from_json,
    save_chunks_to_json,
)


def test_save_and_load_chunks_to_json(tmp_path):
    chunks = [
        {
            "chunk_id": "README.md::chunk_0",
            "source_path": "README.md",
            "source_name": "README.md",
            "source_suffix": ".md",
            "chunk_type": "document",
            "chunk_index": 0,
            "content": "hello world",
            "length": 11,
        }
    ]

    output_path = tmp_path / "chunks.json"

    saved_path = save_chunks_to_json(
        chunks=chunks,
        output_path=str(output_path),
    )

    assert saved_path.exists()

    loaded_chunks = load_chunks_from_json(str(output_path))

    assert loaded_chunks == chunks


def test_load_chunks_from_json_not_exists():
    with pytest.raises(FileNotFoundError):
        load_chunks_from_json("not_exists_chunks.json")


def test_calculate_chunk_stats():
    chunks = [
        {
            "chunk_id": "README.md::chunk_0",
            "source_path": "README.md",
            "source_name": "README.md",
            "source_suffix": ".md",
            "chunk_type": "document",
            "chunk_index": 0,
            "content": "hello",
            "length": 5,
        },
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
    ]

    stats = calculate_chunk_stats(chunks)

    assert stats["total"] == 2
    assert stats["document_count"] == 1
    assert stats["code_count"] == 1
    assert stats["avg_length"] == 10.5


def test_calculate_chunk_stats_empty():
    stats = calculate_chunk_stats([])

    assert stats["total"] == 0
    assert stats["document_count"] == 0
    assert stats["code_count"] == 0
    assert stats["avg_length"] == 0