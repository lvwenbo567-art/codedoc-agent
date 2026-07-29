from dataclasses import dataclass
import asyncio
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
from langgraph_agent.workflow_service import CodeDocWorkflowService


@dataclass
class FakeToolResult:
    success: bool
    data: Any = None
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float = 0.0


class FakeToolExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(
        self,
        tool_name: str,
        arguments: str | dict[str, Any],
    ) -> FakeToolResult:
        self.calls.append(tool_name)

        if tool_name == "search_code":
            return FakeToolResult(
                success=True,
                data={
                    "results": [
                        {
                            "chunk_id": "c-1",
                            "source_path": "app/clients/rerank_client.py",
                            "chunk_type": "code",
                            "content": "class RerankClient: ...",
                            "score": 0.95,
                        }
                    ]
                },
            )

        if tool_name == "search_documents":
            return FakeToolResult(
                success=True,
                data={
                    "results": [
                        {
                            "chunk_id": "d-1",
                            "source_path": "README.md",
                            "chunk_type": "document",
                            "content": "Run with uvicorn.",
                            "score": 0.88,
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
                        },
                        {
                            "path": "tests",
                            "name": "tests",
                            "type": "directory",
                        },
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
            "answer": "这是基于项目证据生成的回答 [Source 1]",
        }


def build_service() -> CodeDocWorkflowService:
    return CodeDocWorkflowService(
        dependencies=CodeDocGraphDependencies(
            tool_executor=FakeToolExecutor(),
            chat_service=FakeChatService(),
        )
    )


def test_code_workflow():
    service = build_service()

    result = service.run(
        query="RerankClient 在哪里定义？",
        project_id=1,
    )

    assert result["query_type"] == "code"
    assert result["evidence_sufficient"] is True
    assert result["execution_steps"] == [
        "initialize",
        "classify_query",
        "code_search",
        "check_evidence",
        "build_answer",
    ]


def test_document_workflow():
    service = build_service()

    result = service.run(
        query="README 中如何启动项目？",
        project_id=1,
    )

    assert result["query_type"] == "document"
    assert "document_search" in result["execution_steps"]


def test_structure_workflow():
    service = build_service()

    result = service.run(
        query="项目有哪些目录和模块？",
        project_id=1,
    )

    assert result["query_type"] == "structure"
    assert "project_structure" in result["execution_steps"]


def test_unknown_workflow():
    service = build_service()

    result = service.run(
        query="今天适合去哪里旅游？",
        project_id=1,
    )

    assert result["query_type"] == "unknown"
    assert result["evidence_sufficient"] is False
    assert result["execution_steps"] == [
        "initialize",
        "classify_query",
        "fallback_answer",
    ]


def test_async_workflow():
    service = build_service()

    result = asyncio.run(
        service.arun(
            query="RerankClient 的代码实现在哪里？",
            project_id=1,
        )
    )

    assert result["query_type"] == "code"
    assert result["answer"] != ""
