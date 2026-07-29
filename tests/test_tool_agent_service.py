from __future__ import annotations

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies
from langgraph_agent.tool_agent_service import CodeDocToolAgentService


class FakeGraph:
    def __init__(self, result_state: dict) -> None:
        self.result_state = result_state
        self.initial_state = None

    def invoke(self, initial_state, config=None):
        self.initial_state = initial_state
        return {
            **initial_state,
            **self.result_state,
        }


def _service(result_state: dict) -> tuple[CodeDocToolAgentService, FakeGraph]:
    runtime = ToolAgentRuntimeConfig(
        project_root=".",
        trace_content_chars=500,
    )
    dependencies = CodeDocToolAgentDependencies(
        runtime=runtime,
        model_config=LangChainModelConfig(provider="mock"),
        model_with_tools=None,
        tools=[],
        allowed_tool_names=frozenset({"search_code", "read_file_range"}),
    )
    graph = FakeGraph(result_state)

    return (
        CodeDocToolAgentService(
            dependencies=dependencies,
            runtime=runtime,
            graph=graph,
        ),
        graph,
    )


def test_initial_state_contains_only_current_human_message() -> None:
    service, graph = _service(
        {
            "answer": "ok",
            "completed": True,
            "stop_reason": "completed",
        }
    )

    result = service.run(query="hello", project_id=1)

    assert result["success"] is True
    assert len(graph.initial_state["messages"]) == 1
    assert isinstance(graph.initial_state["messages"][0], HumanMessage)


def test_result_serializes_ai_tool_and_tool_messages() -> None:
    long_tool_result = "x" * 800
    service, _ = _service(
        {
            "answer": "done",
            "completed": True,
            "stop_reason": "completed",
            "model_call_count": 2,
            "tool_call_count": 1,
            "tool_call_history": [
                {
                    "sequence": 1,
                    "tool_name": "search_code",
                    "arguments": {"query": "x"},
                }
            ],
            "messages": [
                HumanMessage(content="q"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "search_code",
                            "args": {"query": "x"},
                        }
                    ],
                ),
                ToolMessage(
                    content=long_tool_result,
                    tool_call_id="call_1",
                ),
                AIMessage(content="done"),
            ],
        }
    )

    result = service.run(query="q", project_id=1)

    assert result["success"] is True
    assert result["tool_call_history"][0]["tool_name"] == "search_code"
    assert result["message_trace"][1]["tool_calls"][0]["name"] == "search_code"
    assert "[content truncated]" in result["message_trace"][2]["content"]


def test_limited_result_success_is_false() -> None:
    service, _ = _service(
        {
            "answer": "limited",
            "completed": True,
            "stop_reason": "tool_call_limit",
        }
    )

    result = service.run(query="q", project_id=1)

    assert result["success"] is False
    assert result["stop_reason"] == "tool_call_limit"
