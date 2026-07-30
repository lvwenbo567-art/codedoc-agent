from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.checkpoint_config import CheckpointConfig
from langgraph_agent.checkpoint_runtime import (
    CheckpointRuntimeNotStartedError,
    SQLiteCheckpointRuntime,
)
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig


def test_checkpoint_config_resolves_and_creates_parent_directory(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "checkpoints.sqlite"
    config = CheckpointConfig(database_path=str(db_path))

    config.ensure_parent_directory()

    assert Path(config.resolved_database_path).is_absolute()
    assert db_path.parent.exists()


@pytest.mark.asyncio
async def test_runtime_start_is_idempotent_and_close_resets_checkpointer(
    tmp_path: Path,
) -> None:
    runtime = SQLiteCheckpointRuntime(
        config=CheckpointConfig(database_path=str(tmp_path / "cp.sqlite"))
    )

    await runtime.start()
    first_checkpointer = runtime.checkpointer
    await runtime.start()

    assert runtime.started is True
    assert runtime.checkpointer is first_checkpointer

    await runtime.close()

    assert runtime.started is False
    with pytest.raises(CheckpointRuntimeNotStartedError):
        _ = runtime.checkpointer


@pytest.mark.asyncio
async def test_runtime_returns_same_lock_for_same_effective_thread_id(
    tmp_path: Path,
) -> None:
    runtime = SQLiteCheckpointRuntime(
        config=CheckpointConfig(database_path=str(tmp_path / "cp.sqlite"))
    )

    assert runtime.get_thread_lock("project:1:thread:a") is runtime.get_thread_lock(
        "project:1:thread:a"
    )
    assert runtime.get_thread_lock("project:1:thread:a") is not runtime.get_thread_lock(
        "project:1:thread:b"
    )


def test_service_cache_key_changes_when_runtime_changes() -> None:
    model_config = LangChainModelConfig(
        provider="mock",
        model_name="fake",
    )
    first = SQLiteCheckpointRuntime._build_service_cache_key(
        runtime=ToolAgentRuntimeConfig(project_root=".", max_model_calls=3),
        model_config=model_config,
    )
    second = SQLiteCheckpointRuntime._build_service_cache_key(
        runtime=ToolAgentRuntimeConfig(project_root=".", max_model_calls=4),
        model_config=model_config,
    )

    assert first != second
