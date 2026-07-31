from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictAgentQualityModel(BaseModel):
    """
    Agent 质量反馈 API 的严格模型基类。
    """

    model_config = ConfigDict(extra="forbid")


class AgentFeedbackCreateRequest(StrictAgentQualityModel):
    project_id: int = Field(default=1, ge=1)
    thread_id: str = Field(min_length=1, max_length=300)
    run_id: str | None = Field(default=None, max_length=300)
    query: str = Field(min_length=1, max_length=3000)
    answer: str = Field(min_length=1, max_length=12000)
    rating: int = Field(ge=-1, le=1)
    issue_tags: list[str] = Field(default_factory=list)
    comment: str | None = Field(default=None, max_length=3000)
    corrected_answer: str | None = Field(default=None, max_length=12000)


class AgentFeedbackResponse(StrictAgentQualityModel):
    feedback_id: int
    project_id: int
    thread_id: str
    run_id: str | None = None
    query: str
    answer: str
    rating: int
    issue_tags: list[str] = Field(default_factory=list)
    comment: str | None = None
    corrected_answer: str | None = None
    created_at: str


class AgentFeedbackListResponse(StrictAgentQualityModel):
    count: int
    items: list[AgentFeedbackResponse]


class PromoteBadCaseRequest(StrictAgentQualityModel):
    case_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=300)
    expected_tool_names: list[str] = Field(default_factory=list)
    forbidden_tool_names: list[str] = Field(default_factory=list)
    required_answer_terms: list[str] = Field(default_factory=list)
    accepted_stop_reasons: list[str] = Field(
        default_factory=lambda: ["completed"]
    )
    notes: str | None = Field(default=None, max_length=3000)


class BadCaseResponse(StrictAgentQualityModel):
    bad_case_id: int
    feedback_id: int
    case_id: str
    project_id: int
    name: str
    query: str
    expected_tool_names: list[str] = Field(default_factory=list)
    forbidden_tool_names: list[str] = Field(default_factory=list)
    required_answer_terms: list[str] = Field(default_factory=list)
    accepted_stop_reasons: list[str] = Field(default_factory=list)
    notes: str | None = None
    created_at: str
    updated_at: str


class BadCaseListResponse(StrictAgentQualityModel):
    count: int
    items: list[BadCaseResponse]
