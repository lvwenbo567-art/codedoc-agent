from __future__ import annotations

from typing import Any

from repositories.feedback_repository import (
    AgentFeedbackRepository,
    FeedbackNotFoundError,
)


class AgentFeedbackService:
    """
    Feedback 业务服务。

    Repository 负责数据库读写；Service 负责业务语义和校验。
    """

    def __init__(
        self,
        *,
        repository: AgentFeedbackRepository,
    ) -> None:
        self.repository = repository

    @staticmethod
    def _normalize_tags(issue_tags: list[str]) -> list[str]:
        tags: list[str] = []

        for tag in issue_tags:
            value = tag.strip()

            if not value or value in tags:
                continue

            tags.append(value)

        return tags

    def create_feedback(
        self,
        *,
        project_id: int,
        thread_id: str,
        run_id: str | None,
        query: str,
        answer: str,
        rating: int,
        issue_tags: list[str],
        comment: str | None,
        corrected_answer: str | None,
    ) -> dict[str, Any]:
        return self.repository.create_feedback(
            project_id=project_id,
            thread_id=thread_id.strip(),
            run_id=run_id.strip() if run_id else None,
            query=query.strip(),
            answer=answer.strip(),
            rating=rating,
            issue_tags=self._normalize_tags(issue_tags),
            comment=comment.strip() if comment else None,
            corrected_answer=(
                corrected_answer.strip() if corrected_answer else None
            ),
        )

    def list_feedback(
        self,
        *,
        project_id: int | None = None,
        rating: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.repository.list_feedback(
            project_id=project_id,
            rating=rating,
            limit=limit,
            offset=offset,
        )

    def promote_to_bad_case(
        self,
        *,
        feedback_id: int,
        case_id: str,
        name: str,
        expected_tool_names: list[str],
        forbidden_tool_names: list[str],
        required_answer_terms: list[str],
        accepted_stop_reasons: list[str],
        notes: str | None,
    ) -> dict[str, Any]:
        return self.repository.promote_feedback_to_bad_case(
            feedback_id=feedback_id,
            case_id=case_id.strip(),
            name=name.strip(),
            expected_tool_names=self._normalize_tags(expected_tool_names),
            forbidden_tool_names=self._normalize_tags(forbidden_tool_names),
            required_answer_terms=self._normalize_tags(required_answer_terms),
            accepted_stop_reasons=self._normalize_tags(accepted_stop_reasons),
            notes=notes.strip() if notes else None,
        )

    def list_bad_cases(
        self,
        *,
        project_id: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.repository.list_bad_cases(
            project_id=project_id,
            limit=limit,
            offset=offset,
        )


__all__ = [
    "AgentFeedbackService",
    "FeedbackNotFoundError",
]
