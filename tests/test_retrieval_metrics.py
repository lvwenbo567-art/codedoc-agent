from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from evaluation.retrieval_metrics import (
    calculate_hit_at_k,
    calculate_reciprocal_rank,
    extract_chunk_ids,
)


def test_extract_chunk_ids_returns_ordered_ids():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
    ]

    assert extract_chunk_ids(results) == ["a", "b"]


def test_calculate_hit_at_1_hits_expected_chunk():
    results = [
        {"chunk_id": "expected"},
        {"chunk_id": "other"},
    ]

    assert calculate_hit_at_k(results, ["expected"], k=1) == 1.0


def test_calculate_hit_at_3_hits_expected_chunk():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
        {"chunk_id": "expected"},
    ]

    assert calculate_hit_at_k(results, ["expected"], k=3) == 1.0


def test_calculate_hit_at_k_returns_zero_when_not_found():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
    ]

    assert calculate_hit_at_k(results, ["expected"], k=2) == 0.0


def test_calculate_hit_at_k_rejects_invalid_k():
    with pytest.raises(ValueError):
        calculate_hit_at_k([], ["expected"], k=0)


def test_calculate_reciprocal_rank_for_first_result():
    results = [
        {"chunk_id": "expected"},
        {"chunk_id": "other"},
    ]

    assert calculate_reciprocal_rank(results, ["expected"]) == 1.0


def test_calculate_reciprocal_rank_for_second_result():
    results = [
        {"chunk_id": "other"},
        {"chunk_id": "expected"},
    ]

    assert calculate_reciprocal_rank(results, ["expected"]) == 0.5


def test_calculate_reciprocal_rank_returns_zero_when_not_found():
    results = [
        {"chunk_id": "other"},
    ]

    assert calculate_reciprocal_rank(results, ["expected"]) == 0.0
