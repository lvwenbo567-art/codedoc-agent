from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from chunk_utils import (
    calculate_chunk_stats,
    compute_content_hash,
    filter_chunks,
    iter_batches,
)


def build_chunks() -> list[dict]:
    """
    构造 chunk 工具测试数据。
    """
    return [
        {"chunk_type": "code", "content": "def main(): pass"},
        {"chunk_type": "document", "content": "project readme"},
        {"chunk_type": "code", "content": "class App: pass"},
    ]


def test_filter_chunks_by_type():
    results = filter_chunks(build_chunks(), chunk_type="code")

    assert len(results) == 2
    assert all(chunk["chunk_type"] == "code" for chunk in results)


def test_filter_chunks_respects_limit():
    results = filter_chunks(build_chunks(), limit=1)

    assert len(results) == 1


def test_filter_chunks_rejects_negative_limit():
    with pytest.raises(ValueError):
        filter_chunks(build_chunks(), limit=-1)


def test_iter_batches_returns_generator_and_last_batch():
    generator = iter_batches([1, 2, 3], batch_size=2)

    assert iter(generator) is generator
    assert next(generator) == [1, 2]
    assert next(generator) == [3]

    with pytest.raises(StopIteration):
        next(generator)


def test_iter_batches_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        list(iter_batches([1, 2, 3], batch_size=0))


def test_compute_content_hash_ignores_outer_whitespace():
    assert compute_content_hash(" hello ") == compute_content_hash("hello")


def test_compute_content_hash_normalizes_newline_style():
    assert compute_content_hash("a\r\nb") == compute_content_hash("a\nb")


def test_calculate_chunk_stats_counts_types_and_average_length():
    stats = calculate_chunk_stats(build_chunks())

    assert stats["total"] == 3
    assert stats["type_counts"] == {
        "code": 2,
        "document": 1,
    }
    assert stats["average_length"] > 0
