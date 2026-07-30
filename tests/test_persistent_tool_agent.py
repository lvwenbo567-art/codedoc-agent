from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest
from langchain_core.messages import AIMessage


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.checkpoint_config import CheckpointConfig
from langgraph_agent.checkpoint_runtime import SQLiteCheckpointRuntime
from langgraph_agent.persistent_tool_agent_service import (
    PersistentCodeDocToolAgentService,
)
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies
from langgraph_agent.tool_agent_graph import build_codedoc_tool_agent_graph


class MemoryAwareFakeModel:
    def __init__(self) -> None:
        self.message_counts: list[int] = []
        self.human_text_snapshots: list[list[str]] = []

    def invoke(self, messages):
        self.message_counts.append(len(messages))
        self.human_text_snapshots.append(
            [
                str(getattr(message, "content", ""))
                for message in messages
                if message.__class__.__name__ == "HumanMessage"
            ]
        )
        return AIMessage(content=f"answer with {len(messages)} messages")


def _dependencies(model: MemoryAwareFakeModel) -> CodeDocToolAgentDependencies:
    runtime = ToolAgentRuntimeConfig(
        project_root=".",
        max_model_calls=3,
        max_tool_calls=3,
        max_identical_tool_calls=1,
        max_model_messages=20,
        trace_content_chars=1000,
    )

    return CodeDocToolAgentDependencies(
        runtime=runtime,
        model_config=LangChainModelConfig(provider="mock"),
        model_with_tools=model,
        tools=[],
        allowed_tool_names=frozenset(),
    )


async def _build_service(
    *,
    db_path: Path,
    model: MemoryAwareFakeModel,
) -> tuple[SQLiteCheckpointRuntime, PersistentCodeDocToolAgentService]:
    runtime = SQLiteCheckpointRuntime(
        config=CheckpointConfig(database_path=str(db_path))
    )
    await runtime.start()

    dependencies = _dependencies(model)
    graph = build_codedoc_tool_agent_graph(
        dependencies,
        checkpointer=runtime.checkpointer,
    )
    service = PersistentCodeDocToolAgentService(
        dependencies=dependencies,
        graph=graph,
        thread_lock_provider=runtime.get_thread_lock,
    )

    return runtime, service


@pytest.mark.asyncio
async def test_same_thread_accumulates_messages_and_resets_turn_counters(
    tmp_path: Path,
) -> None:
    model = MemoryAwareFakeModel()
    runtime, service = await _build_service(
        db_path=tmp_path / "cp.sqlite",
        model=model,
    )

    try:
        first = await service.arun(
            query="第一轮：记住 keyword_score",
            project_id=1,
            thread_id="same-thread",
        )
        second = await service.arun(
            query="第二轮：它是什么",
            project_id=1,
            thread_id="same-thread",
        )
    finally:
        await runtime.close()

    assert first["turn_index"] == 1
    assert second["turn_index"] == 2
    assert second["message_count"] > first["message_count"]
    assert first["model_call_count"] == 1
    assert second["model_call_count"] == 1
    assert first["tool_call_history"] == []
    assert second["tool_call_history"] == []
    assert model.message_counts[1] > model.message_counts[0]
    assert model.human_text_snapshots[1] == [
        "第一轮：记住 keyword_score",
        "第二轮：它是什么",
    ]


@pytest.mark.asyncio
async def test_different_thread_or_project_does_not_share_messages(
    tmp_path: Path,
) -> None:
    model = MemoryAwareFakeModel()
    runtime, service = await _build_service(
        db_path=tmp_path / "cp.sqlite",
        model=model,
    )

    try:
        await service.arun(
            query="thread A first",
            project_id=1,
            thread_id="thread-a",
        )
        other_thread = await service.arun(
            query="thread B first",
            project_id=1,
            thread_id="thread-b",
        )
        other_project = await service.arun(
            query="project two first",
            project_id=2,
            thread_id="thread-a",
        )
    finally:
        await runtime.close()

    assert other_thread["turn_index"] == 1
    assert other_project["turn_index"] == 1
    assert other_thread["message_count"] == 2
    assert other_project["message_count"] == 2


@pytest.mark.asyncio
async def test_restart_runtime_recovers_history_from_sqlite_file(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cp.sqlite"
    first_model = MemoryAwareFakeModel()
    first_runtime, first_service = await _build_service(
        db_path=db_path,
        model=first_model,
    )

    try:
        first = await first_service.arun(
            query="第一轮持久化",
            project_id=1,
            thread_id="restart-thread",
        )
    finally:
        await first_runtime.close()

    second_model = MemoryAwareFakeModel()
    second_runtime, second_service = await _build_service(
        db_path=db_path,
        model=second_model,
    )

    try:
        second = await second_service.arun(
            query="重启后的第二轮",
            project_id=1,
            thread_id="restart-thread",
        )
    finally:
        await second_runtime.close()

    assert db_path.exists()
    assert first["turn_index"] == 1
    assert second["turn_index"] == 2
    assert second["message_count"] > first["message_count"]
    assert second_model.human_text_snapshots[0] == [
        "第一轮持久化",
        "重启后的第二轮",
    ]


@pytest.mark.asyncio
async def test_same_thread_lock_serializes_requests(tmp_path: Path) -> None:
    model = MemoryAwareFakeModel()
    runtime, service = await _build_service(
        db_path=tmp_path / "cp.sqlite",
        model=model,
    )

    try:
        results = await asyncio.gather(
            service.arun(
                query="并发请求 A",
                project_id=1,
                thread_id="locked-thread",
            ),
            service.arun(
                query="并发请求 B",
                project_id=1,
                thread_id="locked-thread",
            ),
        )
    finally:
        await runtime.close()

    assert sorted(result["turn_index"] for result in results) == [1, 2]
