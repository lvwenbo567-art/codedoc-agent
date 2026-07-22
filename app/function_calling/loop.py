from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from function_calling.client import (
    FunctionCallingClient,
    ModelToolCall,
)
from tools.executor import ToolExecutor
from tools.models import ToolResult
from tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolTrace:
    """
    记录一次工具调用和执行结果。
    """

    tool_call_id: str
    tool_name: str
    arguments: str
    result: ToolResult

    def to_dict(self) -> dict:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result.model_dump(),
        }


@dataclass(frozen=True)
class FunctionCallingLoopResult:
    """
    手写 Function Calling 循环的最终结果。
    """

    query: str
    answer: str
    stop_reason: str
    model_call_count: int
    tool_call_count: int
    tool_traces: list[ToolTrace]
    messages: list[dict[str, Any]]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "model_call_count": self.model_call_count,
            "tool_call_count": self.tool_call_count,
            "tool_traces": [
                trace.to_dict()
                for trace in self.tool_traces
            ],
            "messages": self.messages,
        }


class ManualFunctionCallingLoop:
    """
    手写 Function Calling 循环。

    模型只负责返回工具名和 JSON 参数，
    真正的函数执行由 ToolExecutor 完成。
    """

    def __init__(
        self,
        client: FunctionCallingClient,
        registry: ToolRegistry,
        executor: ToolExecutor,
        max_steps: int = 4,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(
                "max_steps 必须大于 0"
            )

        self.client = client
        self.registry = registry
        self.executor = executor
        self.max_steps = max_steps

    def run(
        self,
        query: str,
    ) -> FunctionCallingLoopResult:
        if not isinstance(query, str):
            raise TypeError(
                "query 必须是字符串"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query 不能为空"
            )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是 CodeDoc Research Agent。"
                    "你可以调用工具查询项目结构、代码和文档。"
                    "如果工具结果不足，请明确说明证据不足。"
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        tool_traces: list[ToolTrace] = []
        model_call_count = 0
        tool_call_count = 0
        answer = ""

        for _ in range(self.max_steps):
            tools = self.registry.to_openai_tools()
            turn = self.client.complete(
                messages=messages,
                tools=tools,
            )
            model_call_count += 1

            if not turn.has_tool_calls():
                answer = (
                    turn.content
                    or "模型未返回最终回答。"
                )
                return FunctionCallingLoopResult(
                    query=query,
                    answer=answer,
                    stop_reason="final_answer",
                    model_call_count=model_call_count,
                    tool_call_count=tool_call_count,
                    tool_traces=tool_traces,
                    messages=messages,
                )

            assistant_message = (
                self._build_assistant_tool_call_message(
                    content=turn.content,
                    tool_calls=turn.tool_calls or [],
                )
            )
            messages.append(assistant_message)

            for tool_call in turn.tool_calls or []:
                result = self.executor.execute(
                    tool_name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
                tool_call_count += 1

                trace = ToolTrace(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                    result=result,
                )
                tool_traces.append(trace)

                messages.append(
                    self._build_tool_result_message(
                        tool_call=tool_call,
                        result=result,
                    )
                )

        final_turn = self.client.complete(
            messages=messages,
            tools=[],
        )
        model_call_count += 1
        answer = (
            final_turn.content
            or "已达到最大工具调用步数，未获得最终回答。"
        )

        return FunctionCallingLoopResult(
            query=query,
            answer=answer,
            stop_reason="max_steps",
            model_call_count=model_call_count,
            tool_call_count=tool_call_count,
            tool_traces=tool_traces,
            messages=messages,
        )

    @staticmethod
    def _build_assistant_tool_call_message(
        *,
        content: str | None,
        tool_calls: list[ModelToolCall],
    ) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in tool_calls
            ],
        }

    @staticmethod
    def _build_tool_result_message(
        *,
        tool_call: ModelToolCall,
        result: ToolResult,
    ) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": json.dumps(
                result.model_dump(),
                ensure_ascii=False,
            ),
        }
