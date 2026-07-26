from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


TraceStatus = Literal[
    "running",
    "completed",
    "failed",
    "limited",
]


class StrictTraceModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )


class ModelCallTrace(StrictTraceModel):#单次模型调用记录
    call_index: int = Field(ge=1)
    started_at: datetime
    completed_at: datetime
    duration_ms: float = Field(ge=0)
    message_count: int = Field(ge=0)#表示调用模型时发送了多少条消息
    available_tool_count: int = Field(ge=0)
    success: bool
    error_type: str | None = None
    error_message: str | None = None


class MiddlewareToolCallTrace(StrictTraceModel):#单次工具调用记录
    call_index: int = Field(ge=1)
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    completed_at: datetime
    duration_ms: float = Field(ge=0)
    success: bool
    result_preview: str = ""
    error_type: str | None = None
    error_message: str | None = None


class MessageTrimTrace(StrictTraceModel):#消息裁剪记录
    trim_index: int = Field(ge=1)
    original_message_count: int = Field(ge=0)
    kept_message_count: int = Field(ge=0)


class AgentRunTrace(StrictTraceModel):
    run_id: str
    trace_id: str
    status: TraceStatus
    started_at: datetime
    completed_at: datetime | None = None
    model_calls: list[ModelCallTrace] = Field(default_factory=list)
    tool_calls: list[MiddlewareToolCallTrace] = Field(default_factory=list)
    message_trims: list[MessageTrimTrace] = Field(default_factory=list)
    degraded: bool = False
    degradation_reasons: list[str] = Field(default_factory=list)
    stop_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
