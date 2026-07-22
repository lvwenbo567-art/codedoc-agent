from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from services.answer_quality import evaluate_answer_quality


def build_citations():
    return [
        {
            "citation_id": "Source 1",
            "source_path": "main.py",
        },
        {
            "citation_id": "Source 2",
            "source_path": "README.md",
        },
    ]


def test_evaluate_answer_quality_valid_citations():
    result = evaluate_answer_quality(
        answer="The project entry is in main.py. [Source 1]",
        citations=build_citations(),
    )

    assert result["is_valid"] is True
    assert result["has_citations"] is True
    assert result["used_citation_ids"] == ["Source 1"]
    assert result["valid_citation_ids"] == ["Source 1"]
    assert result["invalid_citation_ids"] == []
    assert result["valid_citation_rate"] == 1.0
    assert result["warnings"] == []


def test_evaluate_answer_quality_detects_missing_citation_marker():
    result = evaluate_answer_quality(
        answer="The project entry is in main.py.",
        citations=build_citations(),
    )

    assert result["is_valid"] is True
    assert result["has_citations"] is False
    assert result["used_citation_ids"] == []
    assert result["valid_citation_rate"] == 0.0
    assert result["warnings"] == ["回答没有使用任何引用标记"]


def test_evaluate_answer_quality_detects_invalid_source():
    result = evaluate_answer_quality(
        answer="The project entry is in main.py. [Source 3]",
        citations=build_citations(),
    )

    assert result["is_valid"] is False
    assert result["has_citations"] is False
    assert result["invalid_citation_ids"] == ["Source 3"]
    assert result["valid_citation_rate"] == 0.0


def test_evaluate_answer_quality_handles_no_citations():
    result = evaluate_answer_quality(
        answer="No context answer.",
        citations=[],
    )

    assert result["is_valid"] is True
    assert result["has_citations"] is False
    assert result["warnings"] == ["当前没有可用的检索引用"]


def test_evaluate_answer_quality_rejects_empty_answer():
    with pytest.raises(ValueError):
        evaluate_answer_quality(
            answer="   ",
            citations=build_citations(),
        )
