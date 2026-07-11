from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from vector_store import (
    cosine_similarity,
    load_vector_index,
    save_vector_index,
)


def test_cosine_similarity_same_vector():
    score = cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    )

    assert score == 1.0


def test_cosine_similarity_orthogonal_vectors():
    score = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert score == 0.0


def test_cosine_similarity_zero_vector():
    score = cosine_similarity(
        [0.0, 0.0],
        [1.0, 0.0],
    )

    assert score == 0.0


def test_cosine_similarity_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity(
            [1.0, 0.0],
            [1.0],
        )


def test_cosine_similarity_empty_vector():
    with pytest.raises(ValueError):
        cosine_similarity([], [])


def test_save_and_load_vector_index(tmp_path):
    records = [
        {
            "chunk_id": "main.py::chunk_0",
            "embedding": [1.0, 0.0],
        }
    ]

    output_path = tmp_path / "nested" / "vector_index.json"

    saved_path = save_vector_index(
        records=records,
        output_path=str(output_path),
    )

    assert saved_path == output_path
    assert saved_path.exists()

    loaded_records = load_vector_index(str(output_path))

    assert loaded_records == records


def test_load_vector_index_not_exists():
    with pytest.raises(FileNotFoundError):
        load_vector_index("not_exists_vector_index.json")
