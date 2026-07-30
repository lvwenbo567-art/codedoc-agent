from __future__ import annotations

from pathlib import Path
import sys

from langchain_core.messages import AIMessage, ToolMessage


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent import human_review_nodes
from langgraph_agent.human_review_nodes import HumanReviewToolAgentNodes
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies


def _dependencies() -> CodeDocToolAgentDependencies:
    return CodeDocToolAgentDependencies(
        runtime=ToolAgentRuntimeConfig(project_root="."),
        model_config=LangChainModelConfig(provider="mock"),
        model_with_tools=None,
        tools=[],
        allowed_tool_names=frozenset({"search_code", "read_file_range"}),
    )


def _state() -> dict:
    tool_call = {
        "id": "call_1",
        "name": "read_file_range",
        "args": {
            "source_path": "test_project/search.py",
            "start_line": 1,
            "end_line": 10,
        },
    }

    return {
        "query": "读代码",
        "project_id": 1,
        "thread_id": "t1",
        "effective_thread_id": "project:1:thread:t1",
        "messages": [
            AIMessage(
                content="",
                id="ai_1",
                tool_calls=[tool_call],
            )
        ],
        "pending_tool_calls": [tool_call],
        "approval_request_id": "review_1",
        "tool_call_count": 0,
        "max_tool_calls": 10,
        "max_identical_tool_calls": 2,
    }


def test_approve_routes_to_prepare_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        human_review_nodes,
        "interrupt",
        lambda payload: {"decision": "approve"},
    )
    nodes = HumanReviewToolAgentNodes(dependencies=_dependencies())
    command = nodes.human_review_node(_state())

    assert command.goto == "prepare_tools"
    assert command.update["approval_status"] == "approved"


def test_reject_adds_tool_message(monkeypatch) -> None:
    monkeypatch.setattr(
        human_review_nodes,
        "interrupt",
        lambda payload: {"decision": "reject", "feedback": "不允许"},
    )
    nodes = HumanReviewToolAgentNodes(dependencies=_dependencies())
    command = nodes.human_review_node(_state())

    assert command.goto == "agent"
    assert isinstance(command.update["messages"][0], ToolMessage)
    assert command.update["messages"][0].tool_call_id == "call_1"
    assert command.update["approval_status"] == "rejected"


def test_edit_replaces_ai_message_tool_calls(monkeypatch) -> None:
    monkeypatch.setattr(
        human_review_nodes,
        "interrupt",
        lambda payload: {
            "decision": "edit",
            "edited_tool_calls": [
                {
                    "id": "call_1",
                    "name": "read_file_range",
                    "args": {
                        "source_path": "test_project/search.py",
                        "start_line": 1,
                        "end_line": 3,
                    },
                }
            ],
        },
    )
    nodes = HumanReviewToolAgentNodes(dependencies=_dependencies())
    command = nodes.human_review_node(_state())

    edited_message = command.update["messages"][0]

    assert command.goto == "prepare_tools"
    assert edited_message.id == "ai_1"
    assert edited_message.tool_calls[0]["args"]["end_line"] == 3
    assert command.update["approval_status"] == "edited"


def test_edit_changed_tool_call_id_is_invalid(monkeypatch) -> None:
    monkeypatch.setattr(
        human_review_nodes,
        "interrupt",
        lambda payload: {
            "decision": "edit",
            "edited_tool_calls": [
                {"id": "call_2", "name": "read_file_range", "args": {}}
            ],
        },
    )
    nodes = HumanReviewToolAgentNodes(dependencies=_dependencies())
    command = nodes.human_review_node(_state())

    assert command.goto == "limit_answer"
    assert command.update["stop_reason"] == "invalid_review_decision"


def test_edit_unregistered_tool_is_invalid(monkeypatch) -> None:
    monkeypatch.setattr(
        human_review_nodes,
        "interrupt",
        lambda payload: {
            "decision": "edit",
            "edited_tool_calls": [
                {"id": "call_1", "name": "delete_file", "args": {}}
            ],
        },
    )
    nodes = HumanReviewToolAgentNodes(dependencies=_dependencies())
    command = nodes.human_review_node(_state())

    assert command.goto == "limit_answer"
    assert command.update["stop_reason"] == "invalid_review_decision"
