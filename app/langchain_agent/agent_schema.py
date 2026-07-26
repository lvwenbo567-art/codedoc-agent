from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from langchain_agent.trace_schema import AgentRunTrace


AgentStopReason = Literal[
    "completed",
    "model_call_limit",
    "tool_call_limit",
    "recursion_limit",
    "execution_error",
]


class StrictAgentModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )


class LangChainToolTrace(StrictAgentModel):
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    '''
    default_factory=dict 表示：
    如果没有提供 arguments，就为当前对象创建一个新的空字典。
    '''
    success: bool | None = None
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float | None = None
    raw_output: str = ""


class LangChainAgentResult(StrictAgentModel):
    run_id: str | None = None
    trace_id: str | None = None
    project_id: int | None = None
    thread_id: str | None = None
    effective_thread_id: str | None = None
    query: str
    answer: str
    success: bool = True
    degraded: bool = False
    provider: str
    model_name: str
    stop_reason: AgentStopReason
    message_count: int = Field(ge=1)
    history_message_count: int = Field(default=0, ge=0)
    current_turn_message_count: int = Field(default=0, ge=0)
    model_call_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(ge=0)
    message_trim_count: int = Field(default=0, ge=0)
    tool_traces: list[LangChainToolTrace] = Field(default_factory=list)
    trace: AgentRunTrace | None = None
    total_duration_ms: float = Field(ge=0)
