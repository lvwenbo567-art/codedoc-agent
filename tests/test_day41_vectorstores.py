from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from vectorstores.point_id import build_vector_point_id
from vectorstores.models import VectorPoint, VectorSearchFilters
from vectorstores.json_vector_store import JsonVectorStore, cosine_similarity
from services.vector_store_sync_service import sync_vector_index_to_store
from repositories.vector_store import save_vector_index


def make_record(chunk_id: str, embedding: list[float], content: str = "alpha") -> dict:
    return {
        "chunk_id": chunk_id,
        "source_path": "app/demo.py",
        "source_name": "demo.py",
        "source_suffix": ".py",
        "chunk_type": "code",
        "chunk_index": 0,
        "content": content,
        "content_preview": content[:200],
        "length": len(content),
        "embedding": embedding,
        "content_hash": f"hash-{chunk_id}",
        "embedding_model": "mock-hash-embedding",
    }


def test_point_id_is_stable_and_project_scoped():
    first = build_vector_point_id(project_id=1, chunk_id="a.py::chunk_0")
    second = build_vector_point_id(project_id=1, chunk_id="a.py::chunk_0")
    other_project = build_vector_point_id(project_id=2, chunk_id="a.py::chunk_0")

    assert first == second
    assert first != other_project


@pytest.mark.parametrize(
    "project_id,chunk_id",
    [(0, "x"), (1, "")],
)
def test_point_id_rejects_invalid_input(project_id, chunk_id):
    with pytest.raises(ValueError):
        build_vector_point_id(project_id=project_id, chunk_id=chunk_id)


def test_vector_point_from_index_record_strips_vector_from_payload():
    point = VectorPoint.from_index_record(
        project_id=3,
        record=make_record("a", [1, 0], "hello"),
    )

    assert point.project_id == 3
    assert point.chunk_id == "a"
    assert point.dimension == 2
    assert "embedding" not in point.payload
    assert point.payload["content"] == "hello"
    assert point.to_json_record()["embedding"] == [1.0, 0.0]


def test_vector_point_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        VectorPoint.from_index_record(
            project_id=1,
            record=make_record("bad", [float("nan")]),
        )


def test_cosine_similarity_rejects_dimension_mismatch():
    with pytest.raises(Exception, match="维度"):
        cosine_similarity([1, 0], [1])


def test_json_vector_store_upsert_search_filter_and_delete(tmp_path):
    store = JsonVectorStore(index_path=str(tmp_path / "index.json"), project_id=1)
    points = [
        VectorPoint.from_index_record(
            project_id=1,
            record=make_record("a", [1, 0], "alpha function"),
        ),
        VectorPoint.from_index_record(
            project_id=1,
            record={**make_record("b", [0, 1], "beta docs"), "chunk_type": "document"},
        ),
    ]

    upserted = store.upsert(points=points, batch_size=1)

    assert upserted.received_count == 2
    assert upserted.batch_count == 2
    assert store.count(project_id=1) == 2
    assert store.list_chunk_ids(project_id=1) == {"a", "b"}

    results = store.search(
        project_id=1,
        query_vector=[1, 0],
        top_k=2,
        filters=VectorSearchFilters(chunk_type="code"),
    )

    assert [item.chunk_id for item in results] == ["a"]
    assert results[0].to_legacy_record(rank=1)["vector_score"] == results[0].score

    deleted = store.delete_chunks(project_id=1, chunk_ids=["a"])

    assert deleted.deleted_count == 1
    assert store.list_chunk_ids(project_id=1) == {"b"}

    project_deleted = store.delete_project(project_id=1)

    assert project_deleted.deleted_count == 1
    assert store.count(project_id=1) == 0


def test_json_vector_store_rejects_wrong_project(tmp_path):
    store = JsonVectorStore(index_path=str(tmp_path / "index.json"), project_id=1)

    with pytest.raises(ValueError, match="project_id"):
        store.count(project_id=2)


def test_sync_vector_index_to_json_store(tmp_path):
    index_path = tmp_path / "source_index.json"
    store_path = tmp_path / "store.json"
    records = [
        make_record("a", [1, 0], "alpha"),
        make_record("b", [0, 1], "beta"),
    ]
    save_vector_index(
        records=records,
        output_path=str(index_path),
        metadata={
            "index_format_version": "1.0",
            "embedding_provider": "mock",
            "embedding_model": "mock-hash-embedding",
            "dimension": 2,
            "normalized": True,
        },
    )
    store = JsonVectorStore(index_path=str(store_path), project_id=7)

    result = sync_vector_index_to_store(
        project_id=7,
        index_path=str(index_path),
        vector_store=store,
        batch_size=2,
    )

    assert result.project_id == 7
    assert result.source_count == 2
    assert result.upserted_count == 2
    assert result.final_count == 2
    assert result.vector_size == 2
