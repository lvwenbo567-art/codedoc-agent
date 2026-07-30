from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from langgraph_agent.thread_identity import build_effective_thread_id
from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies
from langgraph_agent.tool_agent_service import serialize_agent_messages
from langgraph_agent.tool_agent_state import CodeDocToolAgentState


class PersistentToolAgentExecutionError(RuntimeError):
    """
    Persistent Tool Agent 执行过程中无法恢复的错误。
    """


ThreadLockProvider = Callable[[str], asyncio.Lock]
'''
它表示一个函数类型：
输入 effective_thread_id
返回 asyncio.Lock
'''

class PersistentCodeDocToolAgentService:
    """
    带 SQLite Checkpoint 的 Day38 Tool Agent 服务。

    它每次只提交本轮 HumanMessage，历史 messages 由 LangGraph checkpointer
    根据 configurable.thread_id 自动恢复。
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

    def _build_turn_input(
        self,
        *,
        query: str,
        project_id: int,
        thread_id: str,
        effective_thread_id: str,
        run_id: str,
    ) -> CodeDocToolAgentState:
        """
        构建本轮输入状态。

        注意：messages 里只放本轮 HumanMessage，不手动拼接历史。
        """
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "query 不能为空"
            )

        runtime = self.dependencies.runtime

        return CodeDocToolAgentState(
            query=normalized_query,
            project_id=project_id,
            run_id=run_id,
            thread_id=thread_id,
            effective_thread_id=effective_thread_id,
            messages=[
                HumanMessage(
                    content=normalized_query
                )
            ],
            max_model_calls=runtime.max_model_calls,
            max_tool_calls=runtime.max_tool_calls,
            max_identical_tool_calls=runtime.max_identical_tool_calls,
        )

    def _build_invoke_config(
        self,
        *,
        effective_thread_id: str,
        project_id: int,
        thread_id: str,
        run_id: str,
        recursion_limit: int,
    ) -> dict[str, Any]:
        """
        构建 LangGraph 调用配置。

        configurable.thread_id 是 Checkpointer 找到对应历史状态的关键。
        metadata 只由后端写入，不接收用户自定义 filter key。
        """
        return {
            "configurable": {
                "thread_id": effective_thread_id,
            },
            "metadata": {
                "project_id": project_id,
                "public_thread_id": thread_id,
                "run_id": run_id,
                "agent_type": "codedoc_tool_agent",
            },
            "recursion_limit": recursion_limit,
        }

    async def _ainvoke_graph(
        self,
        *,
        turn_input: CodeDocToolAgentState,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        调用图并要求每个 super-step 同步写入 checkpoint。
        """
        if hasattr(self.graph, "ainvoke"):
            return await self.graph.ainvoke(
                turn_input,
                config=config,
                durability="sync",#每个 super-step 的 checkpoint 写入完成后，再继续下一步
            )

        return await asyncio.to_thread(
            self.graph.invoke,
            turn_input,
            config,
        )

    async def _aget_state(
        self,
        *,
        config: dict[str, Any],
    ) -> Any:
        """
        获取当前 thread 的最新 StateSnapshot。
        """
        if hasattr(self.graph, "aget_state"):
            return await self.graph.aget_state(
                config
            )

        return None

    async def arun(
        self,
        *,
        query: str,
        project_id: int,
        thread_id: str,
        recursion_limit: int = 30,
    ) -> dict[str, Any]:
        """
        执行一次持久化 Agent 调用。
        """
        effective_thread_id = build_effective_thread_id(
            project_id=project_id,
            thread_id=thread_id,
        )
        run_id = f"run_{uuid.uuid4().hex}"

        turn_input = self._build_turn_input(
            query=query,
            project_id=project_id,
            thread_id=thread_id,
            effective_thread_id=effective_thread_id,
            run_id=run_id,
        )

        config = self._build_invoke_config(
            effective_thread_id=effective_thread_id,
            project_id=project_id,
            thread_id=thread_id,
            run_id=run_id,
            recursion_limit=recursion_limit,
        )

        started = time.perf_counter()
        thread_lock = self.thread_lock_provider(
            effective_thread_id
        )

        async with thread_lock:
            try:
                result = await self._ainvoke_graph(
                    turn_input=turn_input,
                    config=config,
                )
            except GraphRecursionError as exc:
                return {
                    "query": query,
                    "project_id": project_id,
                    "thread_id": thread_id,
                    "effective_thread_id": effective_thread_id,
                    "run_id": run_id,
                    "answer": (
                        "Agent 达到 LangGraph recursion_limit，"
                        "本轮执行已停止。"
                    ),
                    "success": False,
                    "completed": True,
                    "stop_reason": "graph_recursion_limit",
                    "turn_index": 0,
                    "model_call_count": 0,
                    "tool_call_count": 0,
                    "message_count": 0,
                    "message_trace": [],
                    "tool_call_history": [],
                    "execution_steps": [
                        "graph_recursion_limit"
                    ],
                    "checkpoint_id": None,
                    "total_duration_ms": round(
                        (time.perf_counter() - started) * 1000,
                        2,
                    ),
                    "error_message": str(exc),
                }
            except Exception as exc:
                raise PersistentToolAgentExecutionError(
                    "Persistent Tool Agent 执行失败："
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            if not isinstance(result, dict):
                raise PersistentToolAgentExecutionError(
                    "Graph 返回结果不是字典"
                )

            snapshot = await self._aget_state(
                config=config
            )

        messages = list(result.get("messages") or [])
        snapshot_config = (
            snapshot.config
            if snapshot is not None
            else {}
        )
        configurable = (
            snapshot_config.get("configurable", {})
            if isinstance(snapshot_config, dict)
            else {}
        )
        stop_reason = str(
            result.get("stop_reason", "execution_error")
        )

        return {
            "query": query,
            "project_id": project_id,
            "thread_id": thread_id,
            "effective_thread_id": effective_thread_id,
            "run_id": run_id,
            "answer": str(result.get("answer") or ""),
            "success": stop_reason == "completed",
            "completed": bool(result.get("completed", False)),
            "stop_reason": stop_reason,
            "turn_index": int(result.get("turn_index", 0)),
            "model_call_count": int(result.get("model_call_count", 0)),
            "tool_call_count": int(result.get("tool_call_count", 0)),
            "message_count": len(messages),
            "message_trace": serialize_agent_messages(
                messages=messages,
                max_content_chars=(
                    self.dependencies.runtime.trace_content_chars
                ),
            ),
            "tool_call_history": list(
                result.get("tool_call_history") or []
            ),
            "execution_steps": list(
                result.get("execution_steps") or []
            ),
            "checkpoint_id": configurable.get("checkpoint_id"),
            "total_duration_ms": round(
                (time.perf_counter() - started) * 1000,
                2,
            ),
            "error_message": result.get("error_message"),
        }
