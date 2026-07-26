from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from langchain_agent.trace_recorder import AgentTraceRecorder, utc_now


def duration_ms(started: float) -> float:
    return round(
        (time.perf_counter() - started) * 1000,
        2,
    )


def preview_value(value: Any, max_chars: int = 2000) -> str:
    if isinstance(value, str):
        return value[:max_chars]

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
        )[:max_chars]
    except TypeError:
        return str(value)[:max_chars]


class CodeDocObservabilityMiddleware:
    """
    记录模型调用和工具调用耗时的项目层可观测性组件。
    """

    def __init__(
        self,
        recorder: AgentTraceRecorder,
    ) -> None:
        self.recorder = recorder

    def record_model_call(
        self,
        *,
        message_count: int,
        available_tool_count: int,
        func: Callable[[], Any],
    ) -> Any:
        started_perf = time.perf_counter()
        started_at = utc_now()

        try:
            result = func()
        except Exception as exc:
            self.recorder.add_model_call(
                started_at=started_at,
                completed_at=utc_now(),
                duration_ms=duration_ms(started_perf),
                message_count=message_count,
                available_tool_count=available_tool_count,
                success=False,
                error=exc,
            )
            raise

        self.recorder.add_model_call(
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=duration_ms(started_perf),
            message_count=message_count,
            available_tool_count=available_tool_count,
            success=True,
        )

        return result

    def record_tool_call(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        func: Callable[[], Any],
    ) -> Any:
        started_perf = time.perf_counter()
        started_at = utc_now()

        try:
            result = func()
        except Exception as exc:
            self.recorder.add_tool_call(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
                started_at=started_at,
                completed_at=utc_now(),
                duration_ms=duration_ms(started_perf),
                success=False,
                result_preview="",
                error=exc,
            )
            raise

        self.recorder.add_tool_call(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=duration_ms(started_perf),
            success=True,
            result_preview=preview_value(result),
        )

        return result
