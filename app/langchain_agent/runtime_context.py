from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CodeDocRuntimeContext:
    """
    单次 Agent 调用的运行时上下文。

    这些信息用于权限、项目隔离和运行配置，不应该拼进聊天消息里。
    """

    user_id: str
    project_id: int
    project_root: str
    chunks_path: str
    index_path: str
    run_id: str
    trace_id: str
    permissions: tuple[str, ...] = field(
        default_factory=lambda: ("project:read",)
    )
    '''
    ("project:read",)

    ("project:read", "file:read")

    ("project:read", "project:write", "file:read")
    '''


def build_effective_thread_id(
    *,
    project_id: int,
    thread_id: str,
) -> str:
    thread_id = thread_id.strip()

    if project_id <= 0:
        raise ValueError("project_id 必须大于 0")

    if not thread_id:
        raise ValueError("thread_id 不能为空")

    return f"project:{project_id}:thread:{thread_id}"
