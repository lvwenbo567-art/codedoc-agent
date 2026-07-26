from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock#导入线程锁，用于避免多个执行线程同时修改同一个 Trace 时互相干扰。
from typing import Any
from uuid import uuid4

from langchain_agent.trace_schema import (
    AgentRunTrace,
    MessageTrimTrace,
    MiddlewareToolCallTrace,
    ModelCallTrace,
    TraceStatus,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgentTraceRecorder:
    """
    单次 Agent 请求的 Trace 记录器。

    每次请求独立创建一个 recorder，避免并发请求互相污染。
    """

    def __init__(
        self,
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        self._lock = Lock()
        self._trace = AgentRunTrace(
            run_id=run_id or f"run_{uuid4().hex}",
            trace_id=trace_id or f"trace_{uuid4().hex}",
            status="running",
            started_at=utc_now(),
        )

    @property#@property 让方法可以像普通属性一样访问
    def run_id(self) -> str:
        return self._trace.run_id

    @property
    def trace_id(self) -> str:
        return self._trace.trace_id

    def add_model_call(
        self,
        *,
        started_at: datetime,
        completed_at: datetime,
        duration_ms: float,
        message_count: int,
        available_tool_count: int,
        success: bool,
        error: Exception | None = None,
    ) -> None:
        with self._lock:
            self._trace.model_calls.append(
                ModelCallTrace(
                    call_index=len(self._trace.model_calls) + 1,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                    message_count=message_count,
                    available_tool_count=available_tool_count,
                    success=success,
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error)[:1000] if error else None,
                )
            )

    def add_tool_call(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        started_at: datetime,
        completed_at: datetime,
        duration_ms: float,
        success: bool,
        result_preview: str = "",
        error: Exception | None = None,
    ) -> None:
        with self._lock:
            self._trace.tool_calls.append(
                MiddlewareToolCallTrace(
                    call_index=len(self._trace.tool_calls) + 1,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                    success=success,
                    result_preview=result_preview[:2000],
                    error_type=type(error).__name__ if error else None,
                    error_message=str(error)[:1000] if error else None,
                )
            )

    def add_message_trim(
        self,
        *,
        original_count: int,
        kept_count: int,
    ) -> None:
        with self._lock:
            self._trace.message_trims.append(
                MessageTrimTrace(
                    trim_index=len(self._trace.message_trims) + 1,
                    original_message_count=original_count,
                    kept_message_count=kept_count,
                )
            )

    def mark_degraded(self, reason: str) -> None:
        reason = reason.strip()

        if not reason:
            return

        with self._lock:
            self._trace.degraded = True

            if reason not in self._trace.degradation_reasons:
                self._trace.degradation_reasons.append(reason)

    def finish(
        self,
        *,
        status: TraceStatus,
        stop_reason: str,
        error: Exception | None = None,
    ) -> None:
        with self._lock:
            self._trace.status = status
            self._trace.stop_reason = stop_reason
            self._trace.completed_at = utc_now()

            if error is not None:
                self._trace.error_type = type(error).__name__
                self._trace.error_message = str(error)[:2000]

    def snapshot(self) -> AgentRunTrace:
        with self._lock:
            return AgentRunTrace.model_validate(
                self._trace.model_dump()
            )
