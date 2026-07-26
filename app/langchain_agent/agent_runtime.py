from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver


@dataclass(slots=True)#表示这个对象只能有声明过的字段，不能随便动态加字段
class AgentRuntime:
    """
    某个项目配置对应的一组 Agent Runtime。s

    同一 Runtime 内复用同一个 Agent 和同一个 Checkpointer，
    才能让相同 thread_id 继续同一段短期记忆。
    """

    agent: Any
    checkpointer: InMemorySaver
    tool_count: int
