from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver#这里导入 LangGraph 的内存版 checkpointer。


def create_in_memory_checkpointer() -> InMemorySaver:
    """
    创建进程内短期记忆 Checkpointer。

    InMemorySaver 只适合本地开发和测试，进程重启后会丢失。
    """
    return InMemorySaver()
