from __future__ import annotations

from typing import Any
'''
CheckpointTuple 是 LangGraph 返回的 checkpoint 数据结构。
它里面包含：
config
checkpoint
metadata
parent_config
'''
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langgraph_agent.thread_identity import build_effective_thread_id
from langgraph_agent.tool_agent_service import serialize_agent_messages


def _extract_channel_values(
    checkpoint_tuple: CheckpointTuple,
) -> dict[str, Any]:
    """
    这个函数负责从 checkpoint 里拿出真正的 State。
    """
    checkpoint = checkpoint_tuple.checkpoint
    values = checkpoint.get("channel_values", {})

    if not isinstance(values, dict):
        return {}

    return values


def _get_checkpoint_id(
    checkpoint_tuple: CheckpointTuple,
) -> str | None:
    """
    从 CheckpointTuple 的 config 中读取 checkpoint_id。
    LangGraph 的 checkpoint_id 通常在：
    checkpoint_tuple.config["configurable"]["checkpoint_id"]
    """
    configurable = checkpoint_tuple.config.get(
        "configurable",
        {},
    )

    if not isinstance(configurable, dict):
        return None

    checkpoint_id = configurable.get("checkpoint_id")

    return str(checkpoint_id) if checkpoint_id is not None else None


def _serialize_state_values(
    values: dict[str, Any],
) -> dict[str, Any]:
    """
    把 checkpoint 里的 State 压缩成 API 可返回的状态摘要。
    """
    messages = list(values.get("messages") or [])

    return {
        "query": values.get("query"),
        "project_id": values.get("project_id"),
        "run_id": values.get("run_id"),
        "thread_id": values.get("thread_id"),
        "effective_thread_id": values.get("effective_thread_id"),
        "turn_index": values.get("turn_index", 0),
        "answer": values.get("answer", ""),
        "completed": values.get("completed", False),
        "stop_reason": values.get("stop_reason", "running"),
        "model_call_count": values.get("model_call_count", 0),
        "tool_call_count": values.get("tool_call_count", 0),
        "execution_steps": list(values.get("execution_steps") or []),
        "tool_call_history": list(values.get("tool_call_history") or []),
        "message_count": len(messages),
        "message_trace": serialize_agent_messages(
            messages=messages,
            max_content_chars=2000,
        ),
        "error_message": values.get("error_message"),
    }


class CheckpointInspectionService:
    """
    查询和删除 LangGraph SQLite Checkpoint 的服务。
    """

    def __init__(
        self,
        *,
        checkpointer: AsyncSqliteSaver,
    ) -> None:
        self.checkpointer = checkpointer

    async def get_latest_state(
        self,
        *,
        project_id: int,
        thread_id: str,
    ) -> dict[str, Any]:
        """
        查询某个 thread 的最新 State。
        """
        effective_thread_id = build_effective_thread_id(
            project_id=project_id,
            thread_id=thread_id,
        )
        config = {
            "configurable": {
                "thread_id": effective_thread_id,
            }
        }
        checkpoint_tuple = await self.checkpointer.aget_tuple(config)
        #去 checkpoint 数据库里找 project:1:thread:day38-demo 这个 thread 的最新 checkpoint。
        if checkpoint_tuple is None:
            return {
                "exists": False,
                "project_id": project_id,
                "thread_id": thread_id,
                "effective_thread_id": effective_thread_id,
                "checkpoint_id": None,
                "created_at": None,
                "state": {},
            }

        values = _extract_channel_values(checkpoint_tuple)

        return {
            "exists": True,
            "project_id": project_id,
            "thread_id": thread_id,
            "effective_thread_id": effective_thread_id,
            "checkpoint_id": _get_checkpoint_id(checkpoint_tuple),
            "created_at": checkpoint_tuple.checkpoint.get("ts"),
            "state": _serialize_state_values(values),
        }

    async def list_history(
        self,
        *,
        project_id: int,
        thread_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        查询某个 thread 的 checkpoint 历史，不接受用户传入 metadata filter。
        """
        effective_thread_id = build_effective_thread_id(
            project_id=project_id,
            thread_id=thread_id,
        )
        config = {
            "configurable": {
                "thread_id": effective_thread_id,
            }
        }
        history: list[dict[str, Any]] = []

        async for checkpoint_tuple in self.checkpointer.alist(
            config,
            limit=limit,
        ):#alist() 会返回这个 thread 的 checkpoint 历史。
            values = _extract_channel_values(checkpoint_tuple)
            metadata = checkpoint_tuple.metadata or {}
            parent_config = checkpoint_tuple.parent_config or {}
            parent_configurable = parent_config.get("configurable", {})

            history.append(
                {
                    "checkpoint_id": _get_checkpoint_id(checkpoint_tuple),
                    "parent_checkpoint_id": (
                        parent_configurable.get("checkpoint_id")
                        if isinstance(parent_configurable, dict)
                        else None
                    ),
                    "created_at": checkpoint_tuple.checkpoint.get("ts"),
                    "source": metadata.get("source"),
                    "step": metadata.get("step"),
                    "run_id": values.get("run_id"),
                    "turn_index": values.get("turn_index", 0),
                    "message_count": len(values.get("messages") or []),
                    "completed": values.get("completed", False),
                    "stop_reason": values.get("stop_reason"),
                    "answer": str(values.get("answer", ""))[:500],
                }
            )

        return history

    async def delete_thread(
        self,
        *,
        project_id: int,
        thread_id: str,
    ) -> dict[str, Any]:
        """
        删除某个 thread 的全部 checkpoints。
        """
        effective_thread_id = build_effective_thread_id(
            project_id=project_id,
            thread_id=thread_id,
        )

        await self.checkpointer.adelete_thread(effective_thread_id)

        return {
            "deleted": True,
            "project_id": project_id,
            "thread_id": thread_id,
            "effective_thread_id": effective_thread_id,
        }
