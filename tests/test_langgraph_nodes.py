from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
        / "app"
    )
)

from langgraph_agent.dependencies import CodeDocGraphDependencies
from langgraph_agent.nodes import CodeDocWorkflowNodes


@dataclass
class FakeToolResult:
    success: bool
    data: Any = None
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float = 0.0


class FakeToolExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(
        self,
        tool_name: str,
        arguments: str | dict[str, Any],
    ) -> FakeToolResult:
        assert isinstance(arguments, dict)
        self.calls.append(
            (
                tool_name,
                arguments,
            )
        )

        if tool_name == "search_code":
            return FakeToolResult(
                success=True,
                data={
                    "results": [
                        {
                            "chunk_id": "code-1",
                            "source_path": "app/example.py",
                            "source_name": "example.py",
                            "chunk_type": "code",
                            "content": "def keyword_score(): ...",
                            "score": 0.91,
                            "start_line": 1,
                            "end_line": 3,
                        }
                    ]
                },
            )

        if tool_name == "get_project_structure":
            return FakeToolResult(
                success=True,
                data={
                    "entries": [
                        {
                            "path": "app",
                            "name": "app",
                            "type": "directory",
                            "depth": 1,
                        },
                        {
                            "path": "app/langgraph_agent",
                            "name": "langgraph_agent",
                            "type": "directory",
                            "depth": 2,
                        },
                        {
                            "path": "app/langgraph_agent/nodes.py",
                            "name": "nodes.py",
                            "type": "file",
                            "depth": 3,
                        }
                    ]
                },
            )

        return FakeToolResult(
            success=False,
            error_code="UNKNOWN_TOOL",
            error_message="unknown",
        )


class FakeChatService:
    def ask(
        self,
        *,
        query: str,
        history: list | None = None,
    ) -> dict:
        return {
            "answer": "基于证据生成回答 [Source 1]",
        }


def build_nodes() -> tuple[CodeDocWorkflowNodes, FakeToolExecutor]:
    executor = FakeToolExecutor()
    nodes = CodeDocWorkflowNodes(
        dependencies=CodeDocGraphDependencies(
            tool_executor=executor,
            chat_service=FakeChatService(),
        )
    )

    return nodes, executor


def test_initialize_and_classify_nodes():
    nodes, _ = build_nodes()

    initialized = nodes.initialize_node(
        {
            "query": "  keyword_score 函数在哪里？ ",
        }
    )

    assert initialized["query"] == "keyword_score 函数在哪里？"
    assert initialized["execution_steps"] == ["initialize"]

    classified = nodes.classify_query_node(
        initialized
    )

    assert classified["query_type"] == "code"
    assert classified["execution_steps"] == ["classify_query"]


def test_code_search_node_converts_tool_result_to_evidence():
    nodes, executor = build_nodes()

    result = nodes.code_search_node(
        {
            "query": "keyword_score 函数在哪里？",
        }
    )

    assert executor.calls[0][0] == "search_code"
    assert result["execution_steps"] == ["code_search"]
    assert result["evidence"][0]["chunk_id"] == "code-1"
    assert result["evidence"][0]["source_path"] == "app/example.py"


def test_project_structure_node_compresses_entries_to_summary_evidence():
    nodes, executor = build_nodes()

    result = nodes.project_structure_node(
        {
            "query": "椤圭洰鏈夊摢浜涗富瑕佺洰褰曪紵",
        }
    )

    assert executor.calls[0][0] == "get_project_structure"
    assert result["execution_steps"] == ["project_structure"]
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["source_path"] == "<project-structure-summary>"
    assert result["evidence"][0]["metadata"]["original_entry_count"] == 3
    assert "app/langgraph_agent" in result["evidence"][0]["content"]


def test_check_evidence_and_build_answer():
    nodes, _ = build_nodes()
    evidence = [
        {
            "chunk_id": "code-1",
            "source_path": "app/example.py",
            "evidence_type": "code_search",
            "content": "def keyword_score(): ...",
            "score": 0.91,
        }
    ]

    checked = nodes.check_evidence_node(
        {
            "evidence": evidence,
        }
    )
    answered = nodes.build_answer_node(
        {
            "query": "keyword_score 函数在哪里？",
            "evidence": evidence,
        }
    )

    assert checked["evidence_sufficient"] is True
    assert answered["answer"] == "基于证据生成回答 [Source 1]"
    assert answered["citations"][0]["citation_id"] == "Source 1"


def test_fallback_answer_node_for_unknown_query():
    nodes, _ = build_nodes()

    result = nodes.fallback_answer_node(
        {
            "query_type": "unknown",
        }
    )

    assert "无法确定" in result["answer"]
    assert result["evidence_sufficient"] is False
