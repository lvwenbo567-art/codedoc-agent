from __future__ import annotations

from pathlib import Path
import sys

from langchain_core.messages import AIMessage


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.human_review_nodes import HumanReviewToolAgentNodes
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies


def _dependencies(
    *,
    enable_human_review: bool = True,
) -> CodeDocToolAgentDependencies:
    return CodeDocToolAgentDependencies(
        runtime=ToolAgentRuntimeConfig(
            project_root=".",
            enable_human_review=enable_human_review,
            approval_required_tools=("read_file_range",),
        ),
        model_config=LangChainModelConfig(provider="mock"),
        model_with_tools=None,
        tools=[],
        allowed_tool_names=frozenset({"search_code", "read_file_range"}),
    )


def _state_for_tool(name: str) -> dict:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": name,
                        "args": {"query": "keyword_score"},
                    }
                ],
            )
        ],
        "stop_reason": "running",
        "tool_call_count": 0,
        "max_tool_calls": 10,
        "max_identical_tool_calls": 2,
    }


def test_no_tool_calls_routes_to_finalize() -> None:
    nodes = HumanReviewToolAgentNodes(dependencies=_dependencies())
    command = nodes.controller_node(
        {
            "messages": [AIMessage(content="answer")],
            "stop_reason": "running",
        }
    )

    assert command.goto == "finalize"


def test_normal_tool_routes_to_prepare_tools() -> None:
    nodes = HumanReviewToolAgentNodes(dependencies=_dependencies())
    command = nodes.controller_node(_state_for_tool("search_code"))

    assert command.goto == "prepare_tools"
    assert command.update["approval_status"] == "not_required"


def test_protected_tool_routes_to_human_review() -> None:
    nodes = HumanReviewToolAgentNodes(dependencies=_dependencies())
    command = nodes.controller_node(_state_for_tool("read_file_range"))

    assert command.goto == "human_review"
    assert command.update["approval_status"] == "pending"
    assert command.update["approval_request_id"]


def test_hitl_disabled_routes_to_prepare_tools() -> None:
    nodes = HumanReviewToolAgentNodes(
        dependencies=_dependencies(enable_human_review=False)
    )
    command = nodes.controller_node(_state_for_tool("read_file_range"))

    assert command.goto == "prepare_tools"


def test_unregistered_tool_routes_to_limit_answer() -> None:
    nodes = HumanReviewToolAgentNodes(dependencies=_dependencies())
    command = nodes.controller_node(_state_for_tool("delete_file"))

    assert command.goto == "limit_answer"
    assert command.update["stop_reason"] == "invalid_tool_call"
