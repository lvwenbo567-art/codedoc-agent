from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import (
    CodeDocToolAgentDependencies,
    build_tool_agent_dependencies,
)
from langgraph_agent.tool_agent_graph import build_codedoc_tool_agent_graph
from langgraph_agent.tool_agent_state import CodeDocToolAgentState


class CodeDocToolAgentExecutionError(RuntimeError):
    pass


def _serialize_message(
    message: Any,
    *,
    max_chars: int,
) -> dict[str, Any]:
    '''这个函数负责把 LangChain Message 转成 JSON 可返回的 dict。'''
    content = str(getattr(message, "content", ""))

    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[content truncated]"

    item: dict[str, Any] = {
        "type": message.__class__.__name__,
        "content": content,
    }

    if isinstance(message, AIMessage):
        item["tool_calls"] = list(message.tool_calls or [])

    if isinstance(message, ToolMessage):
        item["tool_call_id"] = message.tool_call_id
        item["name"] = getattr(message, "name", None)

    return item


def _serialize_messages(
    messages: list[Any],
    *,
    max_chars: int,
) -> list[dict[str, Any]]:
    return [
        _serialize_message(
            message,
            max_chars=max_chars,
        )
        for message in messages
    ]


def serialize_agent_messages(
    messages: list[Any],
    *,
    max_content_chars: int,
) -> list[dict[str, Any]]:
    return _serialize_messages(
        messages,
        max_chars=max_content_chars,
    )


class CodeDocToolAgentService:
    """
    Day37 Tool Agent 服务入口。
    """

    def __init__(
        self,
        *,
        dependencies: CodeDocToolAgentDependencies | None = None,
        runtime: ToolAgentRuntimeConfig | None = None,
        graph: Any | None = None,
    ) -> None:
        self.runtime = runtime or ToolAgentRuntimeConfig()
        self.dependencies = dependencies or build_tool_agent_dependencies(
            runtime=self.runtime,
        )
        self.graph = graph or build_codedoc_tool_agent_graph(
            self.dependencies
        )

    def _build_initial_state(
        self,
        *,
        query: str,
        project_id: int,
    ) -> CodeDocToolAgentState:
        query = query.strip()

        if not query:
            raise ValueError("query 不能为空")

        return {
            "query": query,
            "project_id": project_id,
            "messages": [HumanMessage(content=query)],
            "model_call_count": 0,
            "tool_call_count": 0,
            "max_model_calls": self.runtime.max_model_calls,
            "max_tool_calls": self.runtime.max_tool_calls,
            "max_identical_tool_calls": self.runtime.max_identical_tool_calls,
            "tool_call_history": [],
            "execution_steps": [],
            "answer": "",
            "completed": False,
            "stop_reason": "running",
            "error_message": None,
        }

    def _format_result(
        self,
        *,
        state: CodeDocToolAgentState,
        query: str,
        project_id: int,
    ) -> dict[str, Any]:
        messages = list(state.get("messages") or [])

        return {
            "query": str(state.get("query") or query),
            "project_id": int(state.get("project_id") or project_id),
            "answer": str(state.get("answer") or ""),
            "success": state.get("stop_reason") == "completed",
            "completed": bool(state.get("completed")),
            "stop_reason": state.get("stop_reason", "execution_error"),
            "error_message": state.get("error_message"),
            "model_call_count": int(state.get("model_call_count", 0)),
            "tool_call_count": int(state.get("tool_call_count", 0)),
            "tool_call_history": list(state.get("tool_call_history") or []),
            "execution_steps": list(state.get("execution_steps") or []),
            "message_count": len(messages),
            "message_trace": _serialize_messages(
                messages,
                max_chars=self.runtime.trace_content_chars,
            ),
            "allowed_tools": sorted(self.dependencies.allowed_tool_names),
            "provider": self.dependencies.model_config.provider,
            "model_name": self.dependencies.model_config.model_name,
        }

    def run(
        self,
        *,
        query: str,
        project_id: int = 1,
        recursion_limit: int = 30,
    ) -> dict[str, Any]:
        initial_state = self._build_initial_state(
            query=query,
            project_id=project_id,
        )

        try:
            state = self.graph.invoke(
                initial_state,
                config={"recursion_limit": recursion_limit},
            )
        except GraphRecursionError as exc:
            state = {
                **initial_state,
                "answer": "Agent 达到 Graph recursion_limit，可能存在循环调用。",
                "completed": True,
                "stop_reason": "graph_recursion_limit",
                "error_message": str(exc),
                "execution_steps": [
                    *list(initial_state.get("execution_steps") or []),
                    "graph_recursion_limit",
                ],
            }
        except Exception as exc:
            raise CodeDocToolAgentExecutionError(str(exc)) from exc

        return self._format_result(
            state=state,
            query=query,
            project_id=project_id,
        )

    async def arun(
        self,
        *,
        query: str,
        project_id: int = 1,
        recursion_limit: int = 30,
    ) -> dict[str, Any]:
        initial_state = self._build_initial_state(
            query=query,
            project_id=project_id,
        )

        try:
            if hasattr(self.graph, "ainvoke"):
                state = await self.graph.ainvoke(
                    initial_state,
                    config={"recursion_limit": recursion_limit},
                )
            else:
                state = await asyncio.to_thread(
                    self.graph.invoke,
                    initial_state,
                    {"recursion_limit": recursion_limit},
                )
        except GraphRecursionError as exc:
            state = {
                **initial_state,
                "answer": "Agent 达到 Graph recursion_limit，可能存在循环调用。",
                "completed": True,
                "stop_reason": "graph_recursion_limit",
                "error_message": str(exc),
                "execution_steps": [
                    *list(initial_state.get("execution_steps") or []),
                    "graph_recursion_limit",
                ],
            }
        except Exception as exc:
            raise CodeDocToolAgentExecutionError(str(exc)) from exc

        return self._format_result(
            state=state,
            query=query,
            project_id=project_id,
        )
