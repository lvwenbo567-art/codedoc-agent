from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from repositories.feedback_repository import AgentFeedbackRepository
from schemas.agent_quality_schema import (
    AgentFeedbackCreateRequest,
    AgentFeedbackListResponse,
    AgentFeedbackResponse,
    BadCaseListResponse,
    BadCaseResponse,
    PromoteBadCaseRequest,
)
from services.feedback_service import (
    AgentFeedbackService,
    FeedbackNotFoundError,
)


router = APIRouter(
    prefix="/agent-quality",
    tags=["agent-quality"],
)


def build_feedback_service() -> AgentFeedbackService:
    """
    构建 Feedback 服务。
    """
    return AgentFeedbackService(
        repository=AgentFeedbackRepository()
    )


@router.post(
    "/feedback",
    response_model=AgentFeedbackResponse,
)
def create_feedback(
    body: AgentFeedbackCreateRequest,
) -> AgentFeedbackResponse:
    """
    创建一条用户反馈。
    """
    service = build_feedback_service()
    result = service.create_feedback(
        project_id=body.project_id,
        thread_id=body.thread_id,
        run_id=body.run_id,
        query=body.query,
        answer=body.answer,
        rating=body.rating,
        issue_tags=body.issue_tags,
        comment=body.comment,
        corrected_answer=body.corrected_answer,
    )

    return AgentFeedbackResponse.model_validate(result)


@router.get(
    "/feedback",
    response_model=AgentFeedbackListResponse,
)
def list_feedback(
    project_id: int | None = Query(default=None, ge=1),
    rating: int | None = Query(default=None, ge=-1, le=1),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> AgentFeedbackListResponse:
    """
    查询用户反馈列表。
    """
    service = build_feedback_service()
    items = service.list_feedback(
        project_id=project_id,
        rating=rating,
        limit=limit,
        offset=offset,
    )

    return AgentFeedbackListResponse(
        count=len(items),
        items=[
            AgentFeedbackResponse.model_validate(item)
            for item in items
        ],
    )


@router.post(
    "/feedback/{feedback_id}/bad-case",
    response_model=BadCaseResponse,
)
def promote_feedback_to_bad_case(
    feedback_id: int,
    body: PromoteBadCaseRequest,
) -> BadCaseResponse:
    """
    将一条反馈提升为 Bad Case。
    """
    service = build_feedback_service()

    try:
        result = service.promote_to_bad_case(
            feedback_id=feedback_id,
            case_id=body.case_id,
            name=body.name,
            expected_tool_names=body.expected_tool_names,
            forbidden_tool_names=body.forbidden_tool_names,
            required_answer_terms=body.required_answer_terms,
            accepted_stop_reasons=body.accepted_stop_reasons,
            notes=body.notes,
        )
    except FeedbackNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return BadCaseResponse.model_validate(result)


@router.get(
    "/bad-cases",
    response_model=BadCaseListResponse,
)
def list_bad_cases(
    project_id: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> BadCaseListResponse:
    """
    查询 Bad Case 列表。
    """
    service = build_feedback_service()
    items = service.list_bad_cases(
        project_id=project_id,
        limit=limit,
        offset=offset,
    )

    return BadCaseListResponse(
        count=len(items),
        items=[BadCaseResponse.model_validate(item) for item in items],
    )
