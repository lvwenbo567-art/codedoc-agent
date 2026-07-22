from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from utils.content_hash import compute_content_hash
from services.index_update_service import (
    build_incremental_records,
    build_vector_record,
    validate_reusable_index,
)


class CountingEmbeddingClient:
    """
    测试用 EmbeddingClient，记录实际被向量化的文本数量。
    """

    def __init__(self):
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))

        return [
            [float(len(self.calls)), float(index)]
            for index, _ in enumerate(texts)
        ]


def make_chunk(chunk_id: str, content: str, source_path: str = "file.py") -> dict:
    return {
        "chunk_id": chunk_id,
        "source_path": source_path,
        "source_name": Path(source_path).name,
        "source_suffix": Path(source_path).suffix or ".py",
        "chunk_type": "code",
        "chunk_index": int(chunk_id.rsplit("_", 1)[-1]),
        "content": content,
        "length": len(content),
    }


def test_build_vector_record_adds_content_hash():
    chunk = make_chunk("file.py::chunk_0", "def main(): pass")

    record = build_vector_record(
        chunk=chunk,
        embedding=[1.0, 0.0],
        content_hash="hash-a",
    )

    assert record["chunk_id"] == "file.py::chunk_0"
    assert record["content_hash"] == "hash-a"
    assert record["embedding"] == [1.0, 0.0]


def test_incremental_empty_old_index_all_new():
    chunks = [
        make_chunk("file.py::chunk_0", "alpha"),
        make_chunk("file.py::chunk_1", "beta"),
    ]
    client = CountingEmbeddingClient()

    records, stats = build_incremental_records(
        chunks=chunks,
        old_records=[],
        embedding_client=client,
        batch_size=2,
    )

    assert len(records) == 2
    assert stats["new_count"] == 2
    assert stats["reused_count"] == 0
    assert stats["updated_count"] == 0
    assert stats["deleted_count"] == 0
    assert stats["unique_embedding_count"] == 2
    assert client.calls == [["alpha", "beta"]]


def test_incremental_reuses_unchanged_embedding():
    chunk = make_chunk("file.py::chunk_0", "alpha")
    old_record = build_vector_record(
        chunk=chunk,
        embedding=[9.0, 9.0],
        content_hash=compute_content_hash("alpha"),
    )
    client = CountingEmbeddingClient()

    records, stats = build_incremental_records(
        chunks=[chunk],
        old_records=[old_record],
        embedding_client=client,
        batch_size=2,
    )

    assert records[0]["embedding"] == [9.0, 9.0]
    assert stats["reused_count"] == 1
    assert stats["unique_embedding_count"] == 0
    assert client.calls == []


def test_incremental_detects_updated_new_deleted_and_duplicate_content():
    unchanged_chunk = make_chunk("file.py::chunk_0", "same")
    updated_chunk = make_chunk("file.py::chunk_1", "new text")
    new_chunk = make_chunk("file.py::chunk_2", "duplicate text")
    duplicate_chunk = make_chunk("other.py::chunk_3", "duplicate text", "other.py")

    old_records = [
        build_vector_record(
            chunk=unchanged_chunk,
            embedding=[1.0, 0.0],
            content_hash=compute_content_hash("same"),
        ),
        build_vector_record(
            chunk=make_chunk("file.py::chunk_1", "old text"),
            embedding=[2.0, 0.0],
            content_hash=compute_content_hash("old text"),
        ),
        build_vector_record(
            chunk=make_chunk("deleted.py::chunk_4", "deleted", "deleted.py"),
            embedding=[3.0, 0.0],
            content_hash=compute_content_hash("deleted"),
        ),
    ]
    client = CountingEmbeddingClient()

    records, stats = build_incremental_records(
        chunks=[
            unchanged_chunk,
            updated_chunk,
            new_chunk,
            duplicate_chunk,
        ],
        old_records=old_records,
        embedding_client=client,
        batch_size=2,
    )

    assert [record["chunk_id"] for record in records] == [
        "file.py::chunk_0",
        "file.py::chunk_1",
        "file.py::chunk_2",
        "other.py::chunk_3",
    ]
    assert stats["reused_count"] == 1
    assert stats["updated_count"] == 1
    assert stats["new_count"] == 2
    assert stats["deleted_count"] == 1
    assert stats["deleted_chunk_ids"] == ["deleted.py::chunk_4"]
    assert stats["embedded_chunk_count"] == 3
    assert stats["unique_embedding_count"] == 2
    assert stats["duplicate_content_count"] == 1
    assert client.calls == [["new text", "duplicate text"]]
    assert records[2]["embedding"] == records[3]["embedding"]


def test_validate_reusable_index_rejects_model_change():
    metadata = {
        "index_format_version": "1.0",
        "embedding_provider": "ollama",
        "embedding_model": "bge-m3",
        "normalized": True,
    }

    with pytest.raises(ValueError, match="模型已变化"):
        validate_reusable_index(
            metadata=metadata,
            embedding_provider="ollama",
            embedding_model="other-model",
            normalized=True,
        )


def test_validate_reusable_index_rejects_legacy():
    with pytest.raises(ValueError, match="旧索引没有完整元数据"):
        validate_reusable_index(
            metadata={"index_format_version": "legacy"},
            embedding_provider="mock",
            embedding_model="mock-hash-embedding",
            normalized=True,
        )
