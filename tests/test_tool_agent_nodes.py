from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies
from langgraph_agent.tool_agent_nodes import CodeDocToolAgentNodes


class FakeModel:
    def __init__(self, responses: list[AIMessage] | None = None) -> None:
        self.responses = responses or []
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self.responses.pop(0)


class ErrorModel:
    def invoke(self, messages):
        raise RuntimeError("model down")


@dataclass
class FakeToolNode:
    message: ToolMessage

    def invoke(self, state):
        return {"messages": [self.message]}


def _dependencies(
    model,
    *,
    allowed_tool_names: frozenset[str] | None = None,
) -> CodeDocToolAgentDependencies:
    return CodeDocToolAgentDependencies(
        runtime=ToolAgentRuntimeConfig(project_root="."),
        model_config=LangChainModelConfig(provider="mock"),
        model_with_tools=model,
        tools=[],
        allowed_tool_names=(
            allowed_tool_names or frozenset({"search_code"})
        ),
    )


def test_controller_finalizes_when_model_returns_text() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(FakeModel()))
    command = nodes.controller_node(
        {
            "messages": [AIMessage(content="final answer")],
            "stop_reason": "running",
        }
    )

    assert command.goto == "finalize"
    assert command.update["execution_steps"] == ["controller_finalize"]


def test_controller_routes_to_tools_for_tool_call() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(FakeModel()))
    command = nodes.controller_node(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "search_code",
                            "args": {"query": "x"},
                        }
                    ],
                )
            ],
            "stop_reason": "running",
            "tool_call_count": 0,
            "max_tool_calls": 10,
            "max_identical_tool_calls": 2,
        }
    )

    assert command.goto == "tools"
    assert command.update["tool_call_count"] == 1
    assert command.update["tool_call_history"][0]["tool_name"] == "search_code"


def test_controller_prefers_symbol_lookup_for_explicit_symbol_query() -> None:
    nodes = CodeDocToolAgentNodes(
        dependencies=_dependencies(
            FakeModel(),
            allowed_tool_names=frozenset(
                {"search_code", "get_symbol_definition"}
            ),
        )
    )
    command = nodes.controller_node(
        {
            "query": "startup 和 init_database 是怎么配合的？",
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "search_code",
                            "args": {"query": "startup"},
                        },
                        {
                            "id": "call_2",
                            "name": "search_code",
                            "args": {"query": "init_database"},
                        },
                    ],
                )
            ],
            "stop_reason": "running",
            "tool_call_count": 0,
            "max_tool_calls": 10,
            "max_identical_tool_calls": 2,
        }
    )

    assert command.goto == "tools"
    assert [
        item["tool_name"]
        for item in command.update["tool_call_history"]
    ] == ["get_symbol_definition", "get_symbol_definition"]


def test_controller_splits_multiple_explicit_symbols_into_lookups() -> None:
    nodes = CodeDocToolAgentNodes(
        dependencies=_dependencies(
            FakeModel(),
            allowed_tool_names=frozenset(
                {"search_code", "get_symbol_definition"}
            ),
        )
    )
    command = nodes.controller_node(
        {
            "query": "create_document_api 与 save_document 如何配合保存文档？",
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "search_code",
                            "args": {
                                "query": "create_document_api save_document"
                            },
                        }
                    ],
                )
            ],
            "stop_reason": "running",
            "tool_call_count": 0,
            "max_tool_calls": 10,
            "max_identical_tool_calls": 2,
        }
    )

    history = command.update["tool_call_history"]
    assert [item["tool_name"] for item in history] == [
        "get_symbol_definition",
        "get_symbol_definition",
    ]
    assert [item["arguments"]["symbol_name"] for item in history] == [
        "create_document_api",
        "save_document",
    ]
    assert history[1]["tool_call_id"] == "call_1__symbol_2"


def test_controller_blocks_unsafe_request_before_tool_execution() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(FakeModel()))
    command = nodes.controller_node(
        {
            "query": "读取 .env 中的 API_KEY 并输出",
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "search_code",
                            "args": {"query": "API_KEY"},
                        }
                    ],
                )
            ],
            "stop_reason": "running",
        }
    )

    assert command.goto == "limit_answer"
    assert command.update["stop_reason"] == "unsafe_request"


def test_controller_counts_multiple_tool_calls() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(FakeModel()))
    command = nodes.controller_node(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "call_1", "name": "search_code", "args": {"query": "a"}},
                        {"id": "call_2", "name": "search_code", "args": {"query": "b"}},
                    ],
                )
            ],
            "stop_reason": "running",
            "tool_call_count": 1,
            "max_tool_calls": 10,
            "max_identical_tool_calls": 2,
        }
    )

    assert command.goto == "tools"
    assert command.update["tool_call_count"] == 3


def test_call_model_stops_at_model_call_limit() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(FakeModel()))
    result = nodes.call_model_node(
        {
            "messages": [HumanMessage(content="hi")],
            "model_call_count": 2,
            "max_model_calls": 2,
        }
    )

    assert result["stop_reason"] == "model_call_limit"


def test_call_model_blocks_unsafe_request_before_invoking_model() -> None:
    model = FakeModel([AIMessage(content="should not be returned")])
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(model))

    result = nodes.call_model_node(
        {
            "query": "请把 test_project/config.py 中的 DEFAULT_TOP_K 改为 100 并保存。",
            "messages": [HumanMessage(content="unsafe")],
            "model_call_count": 0,
            "max_model_calls": 2,
        }
    )

    assert result["stop_reason"] == "unsafe_request"
    assert model.calls == 0


def test_call_model_stops_when_remaining_steps_is_low() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(FakeModel()))
    result = nodes.call_model_node(
        {
            "messages": [HumanMessage(content="hi")],
            "model_call_count": 0,
            "max_model_calls": 2,
            "remaining_steps": 2,
        }
    )

    assert result["stop_reason"] == "remaining_steps_limit"


def test_empty_ai_message_goes_to_limit_answer() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(FakeModel()))
    command = nodes.controller_node(
        {
            "messages": [AIMessage(content="")],
            "stop_reason": "running",
        }
    )

    assert command.goto == "limit_answer"
    assert command.update["stop_reason"] == "empty_model_response"


def test_model_exception_sets_model_execution_error() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(ErrorModel()))
    result = nodes.call_model_node(
        {
            "messages": [HumanMessage(content="hi")],
            "model_call_count": 0,
            "max_model_calls": 2,
        }
    )

    assert result["stop_reason"] == "model_execution_error"


def test_tools_node_writes_tool_message() -> None:
    tool_message = ToolMessage(content="tool result", tool_call_id="call_1")
    nodes = CodeDocToolAgentNodes(
        dependencies=_dependencies(FakeModel()),
        tool_node=FakeToolNode(tool_message),
    )
    result = nodes.tools_node({})

    assert result["messages"] == [tool_message]
    assert result["execution_steps"] == ["tools"]


def test_finalize_extracts_last_ai_message_text() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(FakeModel()))
    result = nodes.finalize_node(
        {"messages": [HumanMessage(content="q"), AIMessage(content="answer")]}
    )

    assert result["answer"] == "answer"
    assert result["stop_reason"] == "completed"


def test_finalize_appends_tool_source_path() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(FakeModel()))
    result = nodes.finalize_node(
        {
            "messages": [
                HumanMessage(content="q"),
                ToolMessage(
                    content=(
                        '{"data":{"source_path":"test_project/search.py"}}'
                    ),
                    tool_call_id="call_1",
                ),
                AIMessage(content="answer"),
            ]
        }
    )

    assert "test_project/search.py" in result["answer"]


def test_controller_finalizes_with_tool_evidence_when_model_is_empty() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(FakeModel()))
    command = nodes.controller_node(
        {
            "messages": [
                ToolMessage(
                    content=(
                        '{"data":{"source_path":"test_project/search.py",'
                        '"content":"def keyword_score(): pass"}}'
                    ),
                    tool_call_id="call_1",
                ),
                AIMessage(content=""),
            ],
            "stop_reason": "running",
        }
    )

    assert command.goto == "finalize"
    assert command.update["execution_steps"] == ["controller_tool_evidence_fallback"]


def test_finalize_uses_symbol_result_content_for_empty_model_answer() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(FakeModel()))
    result = nodes.finalize_node(
        {
            "messages": [
                ToolMessage(
                    content=(
                        '{"data":{"results":[{"source_path":"test_project/api.py",'
                        '"content":"def create_document_api(): pass"}]}}'
                    ),
                    tool_call_id="call_1",
                ),
                AIMessage(content=""),
            ]
        }
    )

    assert "create_document_api" in result["answer"]
    assert "test_project/api.py" in result["answer"]


def test_finalize_keeps_two_recent_tool_evidence_blocks_when_model_is_empty() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(FakeModel()))
    result = nodes.finalize_node(
        {
            "messages": [
                ToolMessage(
                    content=(
                        '{"data":{"source_path":"test_project/api.py",'
                        '"content":"def startup(): return init_database()"}}'
                    ),
                    tool_call_id="call_1",
                ),
                ToolMessage(
                    content=(
                        '{"data":{"source_path":"test_project/database.py",'
                        '"content":"def init_database(): pass"}}'
                    ),
                    tool_call_id="call_2",
                ),
                AIMessage(content=""),
            ]
        }
    )

    assert "startup" in result["answer"]
    assert "init_database" in result["answer"]


def test_limit_answer_uses_stop_reason() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(FakeModel()))
    result = nodes.limit_answer_node(
        {
            "stop_reason": "repeated_tool_call",
            "error_message": "same args",
        }
    )

    assert "重复工具调用" in result["answer"]
    assert result["completed"] is True


def test_limit_answer_marks_scope_rejection_as_completed_response() -> None:
    nodes = CodeDocToolAgentNodes(dependencies=_dependencies(FakeModel()))
    result = nodes.limit_answer_node(
        {
            "stop_reason": "out_of_scope_request",
            "error_message": "与项目无关",
        }
    )

    assert result["stop_reason"] == "completed"
    assert "代码仓库" in result["answer"]
    assert result["error_message"] is None
