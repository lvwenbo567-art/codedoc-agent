from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Callable

from langchain_core.tools import BaseTool

from langchain_agent.agent_runtime import AgentRuntime
from langchain_agent.checkpointer_factory import create_in_memory_checkpointer
from langchain_agent.model_config import LangChainModelConfig


def model_config_fingerprint(config: LangChainModelConfig) -> str:
    """
    生成模型配置指纹，避免不同模型共用同一个 Runtime。
    """
    return "|".join(
        [
            config.provider,
            config.model_name,
            config.base_url,
            str(config.temperature),
            str(config.max_tokens),
            str(config.max_retries),
        ]
    )


def build_project_runtime_key(
    *,
    project_id: int,
    project_root: str,
    chunks_path: str,
    index_path: str,
    model_config: LangChainModelConfig,
) -> tuple:
    root = str(Path(project_root).resolve())
    chunks = str(Path(chunks_path).resolve())
    index = str(Path(index_path).resolve())

    return (
        project_id,
        root,
        chunks,
        index,
        model_config_fingerprint(model_config),
    )


class AgentRuntimeCache:
    """
    进程内 Agent Runtime 缓存。

    相同 project_key 复用同一个 Runtime；不同 project_key 隔离。
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._runtimes: dict[tuple, AgentRuntime] = {}

    def clear(self) -> None:
        with self._lock:
            self._runtimes.clear()

    def get_or_create(
        self,
        *,
        key: tuple,
        tools: list[BaseTool],
        create_agent_func: Callable[..., object],
    ) -> AgentRuntime:
        with self._lock:
            existing = self._runtimes.get(key)

            if existing is not None:
                return existing

            checkpointer = create_in_memory_checkpointer()
            agent = create_agent_func(
                checkpointer=checkpointer,
            )
            runtime = AgentRuntime(
                agent=agent,
                checkpointer=checkpointer,
                tool_count=len(tools),
            )
            self._runtimes[key] = runtime

            return runtime


GLOBAL_AGENT_RUNTIME_CACHE = AgentRuntimeCache()
