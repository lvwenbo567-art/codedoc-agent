from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from langgraph_agent.human_review_schema import HumanReviewDecision
from langgraph_agent.thread_identity import build_effective_thread_id
from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies
from langgraph_agent.tool_agent_service import serialize_agent_messages
from langgraph_agent.tool_agent_state import CodeDocToolAgentState


class HumanReviewAgentExecutionError(RuntimeError):
    """
    Human Review Tool Agent 执行失败。
    """


ThreadLockProvider = Callable[[str], asyncio.Lock]


def serialize_interrupt(value: Any) -> dict[str, Any]:
    """
    将 LangGraph Interrupt 对象转换为 API 可返回的 dict。
    """
    interrupt_value = getattr(value, "value", value)

    if isinstance(interrupt_value, dict):
        return interrupt_value

    return {"value": interrupt_value}


def extract_interrupts(result: Any) -> list[dict[str, Any]]:
    """
    从 LangGraph 的执行结果里，把 interrupt(payload) 产生的 payload 提取出来。
    """
    interrupts: list[dict[str, Any]] = []

    if isinstance(result, dict):
        raw_interrupts = result.get("__interrupt__") or []

        if not isinstance(raw_interrupts, list):
            raw_interrupts = [raw_interrupts]

        interrupts.extend(
            serialize_interrupt(item)
            for item in raw_interrupts
        )

    tasks = getattr(result, "tasks", None)

    if tasks:
        for task in tasks:
            for item in getattr(task, "interrupts", []) or []:
                interrupts.append(serialize_interrupt(item))

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in interrupts:
        key = repr(sorted(item.items()))

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


class HumanReviewToolAgentService:
    """
    支持 start / resume 的 Human-in-the-loop Tool Agent 服务。
    """

    def __init__(
        self,
        *,
        dependencies: CodeDocToolAgentDependencies,
        graph: Any,
        thread_lock_provider: ThreadLockProvider,
    ) -> None:
        self.dependencies = dependencies
        self.graph = graph
        self.thread_lock_provider = thread_lock_provider

    def build_config(
        self,
        *,
        project_id: int,
        thread_id: str,
        run_id: str,
        recursion_limit: int,
    ) -> dict[str, Any]:
        effective_thread_id = build_effective_thread_id(
            project_id=project_id,
            thread_id=thread_id,
        )

        return {
            "configurable": {
                "thread_id": effective_thread_id,
            },
            "metadata": {
                "project_id": project_id,
                "public_thread_id": thread_id,
                "run_id": run_id,
                "agent_type": "codedoc_hitl_agent",
            },
            "recursion_limit": recursion_limit,
        }

    def build_turn_input(
        self,
        *,
        query: str,
        project_id: int,
        thread_id: str,
        effective_thread_id: str,
        run_id: str,
    ) -> CodeDocToolAgentState:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query 不能为空")

        runtime = self.dependencies.runtime

        return CodeDocToolAgentState(
            query=normalized_query,
            project_id=project_id,
            run_id=run_id,
            thread_id=thread_id,
            effective_thread_id=effective_thread_id,
            messages=[HumanMessage(content=normalized_query)],
            max_model_calls=runtime.max_model_calls,
            max_tool_calls=runtime.max_tool_calls,
            max_identical_tool_calls=runtime.max_identical_tool_calls,
        )

    async def _ainvoke_graph(
        self,
        graph_input: Any,
        *,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        if hasattr(self.graph, "ainvoke"):
            return await self.graph.ainvoke(
                graph_input,
                config=config,
                durability="sync",
            )

        return await asyncio.to_thread(
            self.graph.invoke,
            graph_input,
            config,
        )

    async def _aget_state(
        self,
        *,
        config: dict[str, Any],
    ) -> Any:
        if hasattr(self.graph, "aget_state"):
            return await self.graph.aget_state(config)

        return None

    def _checkpoint_id_from_snapshot(self, snapshot: Any) -> str | None:
        snapshot_config = getattr(snapshot, "config", None) or {}

        if not isinstance(snapshot_config, dict):
            return None

        configurable = snapshot_config.get("configurable", {})

        if not isinstance(configurable, dict):
            return None

        checkpoint_id = configurable.get("checkpoint_id")

        return str(checkpoint_id) if checkpoint_id is not None else None

    async def _normalize_result(
        self,
        *,
        result: dict[str, Any],
        config: dict[str, Any],
        query: str,
        project_id: int,
        thread_id: str,
        effective_thread_id: str,
        run_id: str,
        started: float,
    ) -> dict[str, Any]:
        snapshot = await self._aget_state(config=config)
        snapshot_values = getattr(snapshot, "values", None)
        values = snapshot_values if isinstance(snapshot_values, dict) else result
        interrupts = [
            *extract_interrupts(result),
            *extract_interrupts(snapshot),
        ]
        messages = list(values.get("messages") or [])
        stop_reason = str(values.get("stop_reason") or "execution_error")
        interrupted = bool(interrupts) or stop_reason == "interrupted"

        if interrupted:
            status = "interrupted"
            success = False
            completed = False
            stop_reason = "interrupted"
        else:
            success = stop_reason == "completed"
            completed = bool(values.get("completed", success))
            status = "completed" if success else "failed"

        return {
            "query": str(values.get("query") or query),
            "project_id": project_id,
            "thread_id": thread_id,
            "effective_thread_id": effective_thread_id,
            "run_id": str(values.get("run_id") or run_id),
            "answer": str(values.get("answer") or ""),
            "status": status,
            "success": success,
            "completed": completed,
            "stop_reason": stop_reason,
            "interrupts": interrupts,
            "approval_status": str(
                values.get("approval_status") or "not_required"
            ),
            "review_history": list(values.get("review_history") or []),
            "turn_index": int(values.get("turn_index", 0)),
            "model_call_count": int(values.get("model_call_count", 0)),
            "tool_call_count": int(values.get("tool_call_count", 0)),
            "message_count": len(messages),
            "message_trace": serialize_agent_messages(
                messages=messages,
                max_content_chars=(
                    self.dependencies.runtime.trace_content_chars
                ),
            ),
            "tool_call_history": list(values.get("tool_call_history") or []),
            "execution_steps": list(values.get("execution_steps") or []),
            "checkpoint_id": self._checkpoint_id_from_snapshot(snapshot),
            "total_duration_ms": round(
                (time.perf_counter() - started) * 1000,
                2,
            ),
            "error_message": values.get("error_message"),
            "allowed_tools": sorted(self.dependencies.allowed_tool_names),
            "provider": self.dependencies.model_config.provider,
            "model_name": self.dependencies.model_config.model_name,
        }

    async def start(
        self,
        *,
        query: str,
        project_id: int,
        thread_id: str,
        recursion_limit: int,
    ) -> dict[str, Any]:
        effective_thread_id = build_effective_thread_id(
            project_id=project_id,
            thread_id=thread_id,
        )
        run_id = f"run_{uuid.uuid4().hex}"
        graph_input = self.build_turn_input(
            query=query,
            project_id=project_id,
            thread_id=thread_id,
            effective_thread_id=effective_thread_id,
            run_id=run_id,
        )
        config = self.build_config(
            project_id=project_id,
            thread_id=thread_id,
            run_id=run_id,
            recursion_limit=recursion_limit,
        )
        started = time.perf_counter()
        lock = self.thread_lock_provider(effective_thread_id)

        async with lock:
            try:
                result = await self._ainvoke_graph(graph_input, config=config)
            except GraphRecursionError as exc:
                result = {
                    **graph_input,
                    "answer": "Agent 达到 LangGraph recursion_limit，执行已停止。",
                    "completed": True,
                    "stop_reason": "graph_recursion_limit",
                    "error_message": str(exc),
                    "execution_steps": ["graph_recursion_limit"],
                }
            except Exception as exc:
                raise HumanReviewAgentExecutionError(str(exc)) from exc

            return await self._normalize_result(
                result=result,
                config=config,
                query=query,
                project_id=project_id,
                thread_id=thread_id,
                effective_thread_id=effective_thread_id,
                run_id=run_id,
                started=started,
            )

    async def resume(
        self,
        *,
        project_id: int,
        thread_id: str,
        decision: HumanReviewDecision,
        recursion_limit: int,
    ) -> dict[str, Any]:
        effective_thread_id = build_effective_thread_id(
            project_id=project_id,
            thread_id=thread_id,
        )
        temporary_run_id = f"resume_{uuid.uuid4().hex}"
        config = self.build_config(
            project_id=project_id,
            thread_id=thread_id,
            run_id=temporary_run_id,
            recursion_limit=recursion_limit,
        )
        started = time.perf_counter()
        lock = self.thread_lock_provider(effective_thread_id)

        async with lock:
            snapshot = await self._aget_state(config=config)

            if snapshot is None:
                raise HumanReviewAgentExecutionError(
                    "没有找到可恢复的 checkpoint"
                )

            values = getattr(snapshot, "values", None) or {}
            run_id = str(values.get("run_id") or temporary_run_id)
            query = str(values.get("query") or "")

            try:
                result = await self._ainvoke_graph(
                    Command(resume=decision.model_dump()),
                    config=config,
                )
            except GraphRecursionError as exc:
                result = {
                    **values,
                    "answer": "Agent 达到 LangGraph recursion_limit，执行已停止。",
                    "completed": True,
                    "stop_reason": "graph_recursion_limit",
                    "error_message": str(exc),
                    "execution_steps": [
                        *list(values.get("execution_steps") or []),
                        "graph_recursion_limit",
                    ],
                }
            except Exception as exc:
                raise HumanReviewAgentExecutionError(str(exc)) from exc

            return await self._normalize_result(
                result=result,
                config=config,
                query=query,
                project_id=project_id,
                thread_id=thread_id,
                effective_thread_id=effective_thread_id,
                run_id=run_id,
                started=started,
            )
