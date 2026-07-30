from __future__ import annotations

import asyncio

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.checkpoint_config import CheckpointConfig
from langgraph_agent.human_review_graph import build_human_review_tool_agent_graph
from langgraph_agent.human_review_service import HumanReviewToolAgentService
from langgraph_agent.persistent_tool_agent_service import (
    PersistentCodeDocToolAgentService,
)
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import build_tool_agent_dependencies
from langgraph_agent.tool_agent_graph import build_codedoc_tool_agent_graph


class CheckpointRuntimeNotStartedError(RuntimeError):
    """
    Checkpoint Runtime 尚未启动。
    """


class SQLiteCheckpointRuntime:
    """
    管理 SQLite Checkpointer、Compiled Graph Service 缓存和 thread 级锁。
    """

    def __init__(
        self,
        *,
        config: CheckpointConfig,
    ) -> None:
        self.config = config
        self._context_manager = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._service_cache: dict[str, PersistentCodeDocToolAgentService] = {}
        self._hitl_service_cache: dict[str, HumanReviewToolAgentService] = {}
        self._thread_locks: dict[str, asyncio.Lock] = {}
        self._runtime_lock = asyncio.Lock()
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def checkpointer(self) -> AsyncSqliteSaver:
        if self._checkpointer is None:
            raise CheckpointRuntimeNotStartedError(
                "Checkpoint Runtime 尚未启动"
            )

        return self._checkpointer

    async def start(self) -> None:
        async with self._runtime_lock:
            if self._started:
                return

            self.config.ensure_parent_directory()
            self._context_manager = AsyncSqliteSaver.from_conn_string(
                self.config.resolved_database_path
            )
            self._checkpointer = await self._context_manager.__aenter__()
            self._started = True

    async def close(self) -> None:
        async with self._runtime_lock:
            if not self._started:
                return

            self._service_cache.clear()
            self._hitl_service_cache.clear()
            self._thread_locks.clear()

            context_manager = self._context_manager
            self._checkpointer = None
            self._context_manager = None
            self._started = False

            if context_manager is not None:
                await context_manager.__aexit__(None, None, None)

    def get_thread_lock(
        self,
        effective_thread_id: str,
    ) -> asyncio.Lock:
        lock = self._thread_locks.get(effective_thread_id)

        if lock is None:
            lock = asyncio.Lock()
            self._thread_locks[effective_thread_id] = lock

        return lock

    @staticmethod
    def _build_service_cache_key(
        *,
        runtime: ToolAgentRuntimeConfig,
        model_config: LangChainModelConfig,
    ) -> str:
        model_key = "|".join(
            [
                str(model_config.provider),
                str(model_config.model_name),
                str(model_config.base_url),
            ]
        )

        return runtime.model_dump_json() + "|" + model_key

    async def get_or_create_service(
        self,
        *,
        runtime: ToolAgentRuntimeConfig,
        model_config: LangChainModelConfig,
    ) -> PersistentCodeDocToolAgentService:
        """
        获取或创建 Day38 持久化 Tool Agent Service。
        """
        if not self._started:
            await self.start()

        cache_key = self._build_service_cache_key(
            runtime=runtime,
            model_config=model_config,
        )
        cached = self._service_cache.get(cache_key)

        if cached is not None:
            return cached

        async with self._runtime_lock:
            cached = self._service_cache.get(cache_key)

            if cached is not None:
                return cached

            dependencies = build_tool_agent_dependencies(
                runtime=runtime,
                model_config=model_config,
            )
            graph = build_codedoc_tool_agent_graph(
                dependencies,
                checkpointer=self.checkpointer,
            )
            service = PersistentCodeDocToolAgentService(
                dependencies=dependencies,
                graph=graph,
                thread_lock_provider=self.get_thread_lock,
            )
            self._service_cache[cache_key] = service

            return service

    async def get_or_create_hitl_service(
        self,
        *,
        runtime: ToolAgentRuntimeConfig,
        model_config: LangChainModelConfig,
    ) -> HumanReviewToolAgentService:
        """
        获取或创建 Day39 Human-in-the-loop Tool Agent Service。
        """
        if not self._started:
            await self.start()

        cache_key = "hitl|" + self._build_service_cache_key(
            runtime=runtime,
            model_config=model_config,
        )
        cached = self._hitl_service_cache.get(cache_key)

        if cached is not None:
            return cached

        async with self._runtime_lock:
            cached = self._hitl_service_cache.get(cache_key)

            if cached is not None:
                return cached

            dependencies = build_tool_agent_dependencies(
                runtime=runtime,
                model_config=model_config,
            )
            invalid_approval_tools = (
                set(runtime.approval_required_tools)
                - set(dependencies.allowed_tool_names)
            )

            if invalid_approval_tools:
                raise ValueError(
                    "approval_required_tools 包含未注册工具："
                    + ", ".join(sorted(invalid_approval_tools))
                )

            graph = build_human_review_tool_agent_graph(
                dependencies,
                checkpointer=self.checkpointer,
            )
            service = HumanReviewToolAgentService(
                dependencies=dependencies,
                graph=graph,
                thread_lock_provider=self.get_thread_lock,
            )
            self._hitl_service_cache[cache_key] = service

            return service
