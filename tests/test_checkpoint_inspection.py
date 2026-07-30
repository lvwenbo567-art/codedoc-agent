from __future__ import annotations

from pathlib import Path
import sys

import pytest
from langchain_core.messages import AIMessage


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.checkpoint_config import CheckpointConfig
from langgraph_agent.checkpoint_inspection import CheckpointInspectionService
from langgraph_agent.checkpoint_runtime import SQLiteCheckpointRuntime
from langgraph_agent.persistent_tool_agent_service import (
    PersistentCodeDocToolAgentService,
)
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies
from langgraph_agent.tool_agent_graph import build_codedoc_tool_agent_graph


class SimpleFakeModel:
    def invoke(self, messages):
        return AIMessage(content="ok")


async def _build_runtime_and_service(
    tmp_path: Path,
) -> tuple[SQLiteCheckpointRuntime, PersistentCodeDocToolAgentService]:
    runtime = SQLiteCheckpointRuntime(
        config=CheckpointConfig(database_path=str(tmp_path / "cp.sqlite"))
    )
    await runtime.start()

    tool_runtime = ToolAgentRuntimeConfig(project_root=".")
    dependencies = CodeDocToolAgentDependencies(
        runtime=tool_runtime,
        model_config=LangChainModelConfig(provider="mock"),
        model_with_tools=SimpleFakeModel(),
        tools=[],
        allowed_tool_names=frozenset(),
    )
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
async def test_latest_state_returns_not_exists_for_missing_thread(
    tmp_path: Path,
) -> None:
    runtime = SQLiteCheckpointRuntime(
        config=CheckpointConfig(database_path=str(tmp_path / "cp.sqlite"))
    )
    await runtime.start()

    try:
        service = CheckpointInspectionService(
            checkpointer=runtime.checkpointer
        )
        result = await service.get_latest_state(
            project_id=1,
            thread_id="missing",
        )
    finally:
        await runtime.close()

    assert result["exists"] is False
    assert result["state"] == {}


@pytest.mark.asyncio
async def test_latest_state_history_limit_and_delete_thread(
    tmp_path: Path,
) -> None:
    runtime, agent_service = await _build_runtime_and_service(tmp_path)

    try:
        await agent_service.arun(
            query="第一轮",
            project_id=1,
            thread_id="inspect-thread",
        )
        await agent_service.arun(
            query="第二轮",
            project_id=1,
            thread_id="inspect-thread",
        )

        inspection = CheckpointInspectionService(
            checkpointer=runtime.checkpointer
        )
        latest = await inspection.get_latest_state(
            project_id=1,
            thread_id="inspect-thread",
        )
        history = await inspection.list_history(
            project_id=1,
            thread_id="inspect-thread",
            limit=3,
        )
        deleted = await inspection.delete_thread(
            project_id=1,
            thread_id="inspect-thread",
        )
        after_delete = await inspection.get_latest_state(
            project_id=1,
            thread_id="inspect-thread",
        )
    finally:
        await runtime.close()

    assert latest["exists"] is True
    assert latest["state"]["turn_index"] == 2
    assert latest["state"]["message_count"] >= 4
    assert len(history) <= 3
    assert history[0]["turn_index"] >= history[-1]["turn_index"]
    assert deleted["deleted"] is True
    assert after_delete["exists"] is False
