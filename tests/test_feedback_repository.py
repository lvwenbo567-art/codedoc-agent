from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from repositories.feedback_repository import (
    AgentFeedbackRepository,
    FeedbackNotFoundError,
)


def create_repository(tmp_path: Path) -> AgentFeedbackRepository:
    return AgentFeedbackRepository(
        db_path=str(tmp_path / "feedback.db")
    )


def create_feedback(
    repository: AgentFeedbackRepository,
    *,
    project_id: int = 1,
    rating: int = -1,
) -> dict:
    return repository.create_feedback(
        project_id=project_id,
        thread_id="thread-1",
        run_id="run-1",
        query="RerankClient 在哪里？",
        answer="错误答案",
        rating=rating,
        issue_tags=["incorrect_answer"],
        comment="应该定位源码",
        corrected_answer="正确答案",
    )


def test_create_and_get_feedback(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    feedback = create_feedback(repository)

    loaded = repository.get_feedback(feedback["feedback_id"])

    assert loaded is not None
    assert loaded["query"] == "RerankClient 在哪里？"
    assert loaded["issue_tags"] == ["incorrect_answer"]


def test_list_feedback_filters_project_id_and_rating(
    tmp_path: Path,
) -> None:
    repository = create_repository(tmp_path)
    create_feedback(repository, project_id=1, rating=-1)
    create_feedback(repository, project_id=2, rating=1)

    by_project = repository.list_feedback(project_id=1)
    by_rating = repository.list_feedback(rating=1)

    assert len(by_project) == 1
    assert by_project[0]["project_id"] == 1
    assert len(by_rating) == 1
    assert by_rating[0]["rating"] == 1


def test_promote_feedback_to_bad_case(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    feedback = create_feedback(repository)

    bad_case = repository.promote_feedback_to_bad_case(
        feedback_id=feedback["feedback_id"],
        case_id="bad-rerank",
        name="Rerank 定位错误",
        expected_tool_names=["get_symbol_definition"],
        forbidden_tool_names=[],
        required_answer_terms=["RerankClient"],
        accepted_stop_reasons=["completed"],
        notes="用户点踩反馈",
    )

    assert bad_case["case_id"] == "bad-rerank"
    assert bad_case["query"] == feedback["query"]
    assert bad_case["expected_tool_names"] == ["get_symbol_definition"]


def test_promote_missing_feedback_raises(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)

    with pytest.raises(FeedbackNotFoundError):
        repository.promote_feedback_to_bad_case(
            feedback_id=999,
            case_id="bad-missing",
            name="Missing",
            expected_tool_names=[],
            forbidden_tool_names=[],
            required_answer_terms=[],
            accepted_stop_reasons=["completed"],
            notes=None,
        )


def test_promote_same_case_id_updates_bad_case(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    feedback = create_feedback(repository)

    first = repository.promote_feedback_to_bad_case(
        feedback_id=feedback["feedback_id"],
        case_id="bad-update",
        name="旧名称",
        expected_tool_names=[],
        forbidden_tool_names=[],
        required_answer_terms=[],
        accepted_stop_reasons=["completed"],
        notes=None,
    )
    second = repository.promote_feedback_to_bad_case(
        feedback_id=feedback["feedback_id"],
        case_id="bad-update",
        name="新名称",
        expected_tool_names=["search_code"],
        forbidden_tool_names=[],
        required_answer_terms=["keyword_score"],
        accepted_stop_reasons=["completed"],
        notes="updated",
    )

    assert first["bad_case_id"] == second["bad_case_id"]
    assert second["name"] == "新名称"
    assert second["expected_tool_names"] == ["search_code"]


def test_delete_feedback_cascades_bad_case(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    feedback = create_feedback(repository)
    repository.promote_feedback_to_bad_case(
        feedback_id=feedback["feedback_id"],
        case_id="bad-delete",
        name="Delete",
        expected_tool_names=[],
        forbidden_tool_names=[],
        required_answer_terms=[],
        accepted_stop_reasons=["completed"],
        notes=None,
    )

    assert repository.delete_feedback(feedback["feedback_id"]) is True
    assert repository.list_bad_cases() == []
