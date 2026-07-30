from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from config import (
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_DEVICE,
    DEFAULT_RERANK_LOCAL_FILES_ONLY,
    DEFAULT_RERANK_MAX_LENGTH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_PROVIDER,
)
from langgraph_agent.human_review_schema import HumanReviewDecision


class StrictHITLAgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HITLAgentBaseRequest(StrictHITLAgentModel):
    project_id: int = Field(ge=1)
    thread_id: str = Field(min_length=1, max_length=120)
    project_root: str = Field(default=".", min_length=1, max_length=1000)
    chunks_path: str = Field(
        default="outputs/chunks.json",
        min_length=1,
        max_length=1000,
    )
    index_path: str = Field(
        default="outputs/vector_index.json",
        min_length=1,
        max_length=1000,
    )
    max_model_calls: int = Field(default=6, ge=1, le=20)
    max_tool_calls: int = Field(default=10, ge=1, le=50)
    max_identical_tool_calls: int = Field(default=2, ge=1, le=5)
    max_model_messages: int = Field(default=18, ge=4, le=100)
    trace_content_chars: int = Field(default=3000, ge=500, le=12000)
    recursion_limit: int = Field(default=40, ge=8, le=120)
    enable_human_review: bool = True
    approval_required_tools: list[str] = Field(
        default_factory=lambda: ["read_file_range"]
    )
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        min_length=1,
        max_length=200,
    )
    embedding_base_url: str = Field(
        default=DEFAULT_EMBEDDING_BASE_URL,
        max_length=1000,
    )
    embedding_api_key: str = Field(
        default=DEFAULT_EMBEDDING_API_KEY,
        max_length=1000,
    )
    embedding_timeout_seconds: float = Field(
        default=DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
        gt=0,
    )
    mock_dimension: int = Field(default=DEFAULT_EMBEDDING_DIMENSION, gt=0)
    rerank_provider: str = DEFAULT_RERANK_PROVIDER
    rerank_model: str = Field(default=DEFAULT_RERANK_MODEL, max_length=1000)
    rerank_device: str = Field(default=DEFAULT_RERANK_DEVICE, max_length=100)
    rerank_batch_size: int = Field(default=DEFAULT_RERANK_BATCH_SIZE, gt=0)
    rerank_max_length: int = Field(default=DEFAULT_RERANK_MAX_LENGTH, gt=0)
    rerank_local_files_only: bool = DEFAULT_RERANK_LOCAL_FILES_ONLY


class HITLAgentStartRequest(HITLAgentBaseRequest):
    query: str = Field(min_length=1, max_length=3000)


class HITLAgentResumeRequest(HITLAgentBaseRequest):
    decision: HumanReviewDecision


class HITLAgentResponse(StrictHITLAgentModel):
    query: str
    project_id: int
    thread_id: str
    effective_thread_id: str
    run_id: str
    answer: str
    status: str
    success: bool
    completed: bool
    stop_reason: str
    interrupts: list[dict[str, Any]] = Field(default_factory=list)
    approval_status: str
    review_history: list[dict[str, Any]] = Field(default_factory=list)
    turn_index: int
    model_call_count: int
    tool_call_count: int
    message_count: int
    message_trace: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_history: list[dict[str, Any]] = Field(default_factory=list)
    execution_steps: list[str] = Field(default_factory=list)
    checkpoint_id: str | None = None
    total_duration_ms: float
    error_message: str | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    provider: str
    model_name: str
