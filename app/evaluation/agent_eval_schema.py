from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictEvalModel(BaseModel):
    """
    Day40 评测相关模型基类。

    extra="forbid" 可以防止评测集字段写错后被静默忽略。
    """

    model_config = ConfigDict(extra="forbid")


class AgentEvalCase(StrictEvalModel):
    """
    一条 Agent 评测样例。

    expected_tool_names 用来衡量工具召回；
    forbidden_tool_names 用来检查安全边界；
    required_answer_terms 用来粗粒度检查答案事实覆盖。
    """

    case_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=300)
    query: str = Field(min_length=1, max_length=3000)
    project_id: int = Field(default=1, ge=1)
    expected_status: Literal["completed", "interrupted"] = "completed"
    expected_tool_names: list[str] = Field(default_factory=list)
    forbidden_tool_names: list[str] = Field(default_factory=list)
    expected_first_tool: str | None = None
    required_answer_terms: list[str] = Field(default_factory=list)
    accepted_stop_reasons: list[str] = Field(
        default_factory=lambda: ["completed"]
    )
    max_latency_ms: float | None = Field(default=None, gt=0)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None


class AgentEvalObservation(StrictEvalModel):
    """
    Agent 执行后抽取出来的观测结果。
    """

    status: Literal["completed", "interrupted", "failed"]
    answer: str = ""
    tool_names: list[str] = Field(default_factory=list)
    stop_reason: str
    latency_ms: float = Field(ge=0)
    error_message: str | None = None
    raw_output: dict[str, Any] = Field(default_factory=dict)


class AgentEvalCaseResult(StrictEvalModel):
    case_id: str
    name: str
    passed: bool
    task_success: float
    tool_exact_match: float
    tool_precision: float#实际调用的工具里，有多少是预期的
    tool_recall: float# 预期工具有没有都用到
    tool_f1: float# precision 和 recall 的综合
    first_tool_accuracy: float
    forbidden_tool_safety: float
    answer_term_coverage: float
    completion_score: float
    latency_pass: float
    latency_ms: float
    expected_tools: list[str]
    actual_tools: list[str]
    missing_tools: list[str]
    unexpected_tools: list[str]
    forbidden_tools_used: list[str]
    missing_answer_terms: list[str]
    failure_reasons: list[str]
    observation: AgentEvalObservation


class AgentEvalSummary(StrictEvalModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    task_success_rate: float
    tool_exact_match_rate: float
    average_tool_precision: float
    average_tool_recall: float
    average_tool_f1: float
    first_tool_accuracy: float
    forbidden_tool_safety_rate: float
    average_answer_term_coverage: float
    completion_rate: float
    latency_pass_rate: float
    average_latency_ms: float
    p95_latency_ms: float


class AgentEvalReport(StrictEvalModel):
    evaluation_id: str
    generated_at: str
    dataset_path: str
    model_provider: str
    model_name: str
    summary: AgentEvalSummary
    results: list[AgentEvalCaseResult]
