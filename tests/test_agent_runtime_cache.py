from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from langchain_agent.agent_runtime_cache import (
    AgentRuntimeCache,
    build_project_runtime_key,
)
from langchain_agent.model_config import LangChainModelConfig


def test_agent_runtime_cache_reuses_same_project_key():
    cache = AgentRuntimeCache()
    key = ("project", 1)

    first = cache.get_or_create(
        key=key,
        tools=[],
        create_agent_func=lambda checkpointer: object(),
    )
    second = cache.get_or_create(
        key=key,
        tools=[],
        create_agent_func=lambda checkpointer: object(),
    )

    assert first is second
    assert first.checkpointer is second.checkpointer


def test_agent_runtime_cache_isolates_different_project_key():
    cache = AgentRuntimeCache()

    first = cache.get_or_create(
        key=("project", 1),
        tools=[],
        create_agent_func=lambda checkpointer: object(),
    )
    second = cache.get_or_create(
        key=("project", 2),
        tools=[],
        create_agent_func=lambda checkpointer: object(),
    )

    assert first is not second
    assert first.checkpointer is not second.checkpointer


def test_project_runtime_key_contains_project_id_and_model_config(tmp_path):
    model_config = LangChainModelConfig(
        provider="openai_compatible",
        model_name="qwen3.5:4b",
        base_url="http://localhost:11434/v1",
    )

    key = build_project_runtime_key(
        project_id=7,
        project_root=str(tmp_path),
        chunks_path=str(tmp_path / "chunks.json"),
        index_path=str(tmp_path / "index.json"),
        model_config=model_config,
    )

    assert key[0] == 7
    assert "qwen3.5:4b" in key[-1]
