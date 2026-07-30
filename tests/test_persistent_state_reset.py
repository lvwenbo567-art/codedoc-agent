from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langgraph.types import Overwrite

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies
from langgraph_agent.tool_agent_nodes import CodeDocToolAgentNodes


@dataclass
class FakeModel:
    def invoke(self, messages):
        return None


def _nodes() -> CodeDocToolAgentNodes:
    runtime = ToolAgentRuntimeConfig(project_root=".")
    dependencies = CodeDocToolAgentDependencies(
        runtime=runtime,
        model_config=LangChainModelConfig(provider="mock"),
        model_with_tools=FakeModel(),
        tools=[],
        allowed_tool_names=frozenset(),
    )

    return CodeDocToolAgentNodes(dependencies=dependencies)


def test_initialize_resets_per_turn_fields_with_overwrite() -> None:
    nodes = _nodes()

    result = nodes.initialize_node(
        {
            "query": " second turn ",
            "turn_index": 1,
            "model_call_count": 9,
            "tool_call_count": 8,
            "tool_call_history": [
                {"tool_name": "search_code"},
            ],
            "execution_steps": ["old_step"],
            "answer": "old answer",
            "completed": True,
            "stop_reason": "completed",
            "error_message": "old error",
        }
    )

    assert result["query"] == "second turn"
    assert result["turn_index"] == 2
    assert result["model_call_count"] == 0
    assert result["tool_call_count"] == 0
    assert isinstance(result["tool_call_history"], Overwrite)
    assert result["tool_call_history"].value == []
    assert isinstance(result["execution_steps"], Overwrite)
    assert result["execution_steps"].value == ["initialize"]
    assert result["answer"] == ""
    assert result["completed"] is False
    assert result["stop_reason"] == "running"
    assert result["error_message"] is None
