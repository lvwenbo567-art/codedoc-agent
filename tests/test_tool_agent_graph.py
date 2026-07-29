from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies
from langgraph_agent.tool_agent_graph import build_codedoc_tool_agent_graph


class SequenceModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = responses

    def invoke(self, messages):
        if not self.responses:
            raise AssertionError("No fake model responses left")

        return self.responses.pop(0)


@dataclass
class EchoToolNode:
    def invoke(self, state):
        last_ai = next(
            message
            for message in reversed(state["messages"])
            if isinstance(message, AIMessage)
        )
        tool_messages = [
            ToolMessage(
                content=f"result for {tool_call['name']}",
                tool_call_id=tool_call["id"],
            )
            for tool_call in last_ai.tool_calls
        ]

        return {"messages": tool_messages}


def _dependencies(
    responses: list[AIMessage],
    *,
    max_tool_calls: int = 10,
    max_identical_tool_calls: int = 2,
) -> CodeDocToolAgentDependencies:
    return CodeDocToolAgentDependencies(
        runtime=ToolAgentRuntimeConfig(
            project_root=".",
            max_tool_calls=max_tool_calls,
            max_identical_tool_calls=max_identical_tool_calls,
        ),
        model_config=LangChainModelConfig(provider="mock"),
        model_with_tools=SequenceModel(responses),
        tools=[],
        allowed_tool_names=frozenset({"search_code", "search_documents"}),
    )


def _tool_call(
    *,
    call_id: str,
    name: str = "search_code",
    query: str = "keyword_score",
) -> dict:
    return {
        "id": call_id,
        "name": name,
        "args": {"query": query},
    }


def test_graph_runs_model_tool_model_finalize_loop() -> None:
    graph = build_codedoc_tool_agent_graph(
        _dependencies(
            [
                AIMessage(content="", tool_calls=[_tool_call(call_id="call_1")]),
                AIMessage(content="final answer"),
            ]
        ),
        tool_node=EchoToolNode(),
    )

    result = graph.invoke(
        {
            "query": "where",
            "project_id": 1,
            "messages": [{"role": "user", "content": "where"}],
            "execution_steps": [],
        },
        config={"recursion_limit": 20},
    )

    assert result["stop_reason"] == "completed"
    assert result["answer"] == "final answer"
    assert result["model_call_count"] == 2
    assert result["tool_call_count"] == 1
    assert result["execution_steps"] == [
        "initialize",
        "model_call",
        "controller_tools",
        "tools",
        "model_call",
        "controller_finalize",
        "finalize",
    ]


def test_graph_blocks_repeated_identical_tool_call() -> None:
    graph = build_codedoc_tool_agent_graph(
        _dependencies(
            [
                AIMessage(content="", tool_calls=[_tool_call(call_id="call_1")]),
                AIMessage(content="", tool_calls=[_tool_call(call_id="call_2")]),
                AIMessage(content="", tool_calls=[_tool_call(call_id="call_3")]),
            ],
            max_identical_tool_calls=2,
        ),
        tool_node=EchoToolNode(),
    )

    result = graph.invoke(
        {
            "query": "repeat",
            "project_id": 1,
            "messages": [{"role": "user", "content": "repeat"}],
            "execution_steps": [],
        },
        config={"recursion_limit": 30},
    )

    assert result["stop_reason"] == "repeated_tool_call"
    assert result["completed"] is True


def test_graph_allows_different_tool_calls() -> None:
    graph = build_codedoc_tool_agent_graph(
        _dependencies(
            [
                AIMessage(content="", tool_calls=[_tool_call(call_id="call_1")]),
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call(
                            call_id="call_2",
                            name="search_documents",
                            query="README",
                        )
                    ],
                ),
                AIMessage(content="done"),
            ],
        ),
        tool_node=EchoToolNode(),
    )

    result = graph.invoke(
        {
            "query": "different",
            "project_id": 1,
            "messages": [{"role": "user", "content": "different"}],
            "execution_steps": [],
        },
        config={"recursion_limit": 30},
    )

    assert result["stop_reason"] == "completed"
    assert result["tool_call_count"] == 2


def test_graph_blocks_tool_call_budget() -> None:
    graph = build_codedoc_tool_agent_graph(
        _dependencies(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call(call_id="call_1"),
                        _tool_call(call_id="call_2", query="another"),
                    ],
                )
            ],
            max_tool_calls=1,
        ),
        tool_node=EchoToolNode(),
    )

    result = graph.invoke(
        {
            "query": "budget",
            "project_id": 1,
            "messages": [{"role": "user", "content": "budget"}],
            "execution_steps": [],
        },
        config={"recursion_limit": 20},
    )

    assert result["stop_reason"] == "tool_call_limit"


def test_graph_recursion_limit_can_be_raised_by_langgraph() -> None:
    graph = build_codedoc_tool_agent_graph(
        _dependencies(
            [
                AIMessage(content="", tool_calls=[_tool_call(call_id="call_1")]),
                AIMessage(content="", tool_calls=[_tool_call(call_id="call_2", query="x2")]),
            ],
        ),
        tool_node=EchoToolNode(),
    )

    with pytest.raises(Exception):
        graph.invoke(
            {
                "query": "low recursion",
                "project_id": 1,
                "messages": [{"role": "user", "content": "low recursion"}],
                "execution_steps": [],
            },
            config={"recursion_limit": 3},
        )
