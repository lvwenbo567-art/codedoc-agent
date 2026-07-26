from __future__ import annotations

from langchain_core.messages import BaseMessage

from langchain_agent.runtime_context import build_effective_thread_id


def resolve_effective_thread_id(
    *,
    project_id: int,
    thread_id: str,
) -> str:
    """
    将前端 thread_id 转成项目隔离后的真实 thread_id。
    """
    return build_effective_thread_id(
        project_id=project_id,
        thread_id=thread_id,
    )


def count_current_turn_messages(
    *,
    before_count: int,
    after_messages: list[BaseMessage],
) -> int:
    """
    根据调用前后消息数量估算本轮新增消息数。
    """
    return max(
        0,
        len(after_messages) - before_count,
    )
