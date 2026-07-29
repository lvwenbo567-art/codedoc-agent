from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langgraph_agent.decision_schema import EvidenceAssessment, QueryDecision
from langgraph_agent.dependencies import CodeDocGraphDependencies
from langgraph_agent.graph import build_codedoc_agentic_rag_graph
from langgraph_agent.rag_runtime import RAGRuntimeConfig


@dataclass
class FakeToolResult:
    success: bool
    data: Any = None
    error_message: str | None = None


class FakeToolExecutor:
    def __init__(self, *, symbol_hit: bool = True, has_evidence: bool = True) -> None:
        self.symbol_hit = symbol_hit
        self.has_evidence = has_evidence

    def execute(self, tool_name: str, arguments: dict | str) -> FakeToolResult:
        if tool_name == "get_symbol_definition":
            return FakeToolResult(
                success=True,
                data={
                    "result_count": 1 if self.symbol_hit else 0,
                    "results": [
                        {
                            "chunk_id": "symbol-1",
                            "source_path": "app/a.py",
                            "symbol_name": "retrieve_with_rerank",
                            "content": "def retrieve_with_rerank(): pass",
                        }
                    ]
                    if self.symbol_hit
                    else [],
                },
            )

        if tool_name in {"search_code", "search_documents"}:
            return FakeToolResult(
                success=True,
                data={
                    "query_strategy": arguments["query_strategy"],
                    "rerank_applied": True,
                    "results": [
                        {
                            "chunk_id": "chunk-1",
                            "source_path": "app/a.py",
                            "chunk_type": "code",
                            "content": "def a(): pass",
                            "score": 0.8,
                        }
                    ]
                    if self.has_evidence
                    else [],
                },
            )

        if tool_name == "get_project_structure":
            return FakeToolResult(
                success=True,
                data={"entries": [{"path": "app", "type": "directory"}]},
            )

        return FakeToolResult(success=False, error_message="unknown")


class FakeDecisionService:
    def __init__(self, decision: QueryDecision) -> None:
        self.decision = decision

    def analyze(self, query: str) -> QueryDecision:
        return self.decision


class FakeEvidenceQualityService:
    def __init__(self, sufficient: bool = True) -> None:
        self.sufficient = sufficient

    def assess(self, **kwargs) -> EvidenceAssessment:
        return EvidenceAssessment(
            sufficient=self.sufficient,
            relevance_score=0.9 if self.sufficient else 0.1,
            coverage_score=0.9 if self.sufficient else 0.1,
            reason="ok" if self.sufficient else "missing",
            missing_information=[] if self.sufficient else ["missing"],
            assessment_method="rule",
        )


class FakeAnswerService:
    def build_answer(self, *, query: str, evidence: list[dict]) -> dict:
        return {
            "answer": "answer [Source 1]",
            "citations": [{"citation_id": "Source 1", "source_path": "app/a.py"}],
            "answer_quality": {"is_valid": True},
        }


class FakeChatService:
    def ask(self, *, query: str, history: list | None = None) -> dict:
        return {"answer": "chat answer"}


def run_graph(
    *,
    decision: QueryDecision,
    symbol_hit: bool = True,
    sufficient: bool = True,
    has_evidence: bool = True,
) -> dict:
    dependencies = CodeDocGraphDependencies(
        tool_executor=FakeToolExecutor(
            symbol_hit=symbol_hit,
            has_evidence=has_evidence,
        ),
        chat_service=FakeChatService(),
        query_decision_service=FakeDecisionService(decision),
        evidence_quality_service=FakeEvidenceQualityService(sufficient=sufficient),
        answer_service=FakeAnswerService(),
        runtime=RAGRuntimeConfig(),
    )
    graph = build_codedoc_agentic_rag_graph(dependencies)

    return graph.invoke(
        {
            "query": "test query",
            "project_id": 1,
            "evidence": [],
            "execution_steps": [],
            "degrade_reasons": [],
        }
    )


def test_exact_symbol_success_path():
    result = run_graph(
        decision=QueryDecision(
            query_type="code",
            retrieval_strategy="original",
            symbol_name="retrieve_with_rerank",
            confidence=0.9,
            reason="test",
        )
    )

    assert result["execution_steps"] == [
        "initialize",
        "analyze_query",
        "exact_symbol_lookup",
        "assess_evidence",
        "build_answer",
    ]


def test_exact_symbol_miss_falls_back_to_code_retrieve():
    result = run_graph(
        symbol_hit=False,
        decision=QueryDecision(
            query_type="code",
            retrieval_strategy="original",
            symbol_name="missing_symbol",
            confidence=0.9,
            reason="test",
        ),
    )

    assert "exact_symbol_lookup" in result["execution_steps"]
    assert "code_retrieve" in result["execution_steps"]


def test_natural_language_code_multi_query_path():
    result = run_graph(
        decision=QueryDecision(
            query_type="code",
            retrieval_strategy="multi_query",
            symbol_name=None,
            confidence=0.8,
            reason="test",
        )
    )

    assert result["execution_steps"] == [
        "initialize",
        "analyze_query",
        "code_retrieve",
        "assess_evidence",
        "build_answer",
    ]
    assert result["retrieval_metadata"]["query_strategy"] == "multi_query"


def test_document_path_and_insufficient_path():
    document_result = run_graph(
        decision=QueryDecision(
            query_type="document",
            retrieval_strategy="multi_query",
            symbol_name=None,
            confidence=0.8,
            reason="test",
        )
    )
    insufficient_result = run_graph(
        sufficient=False,
        has_evidence=False,
        decision=QueryDecision(
            query_type="code",
            retrieval_strategy="multi_query",
            symbol_name=None,
            confidence=0.8,
            reason="test",
        ),
    )

    assert "document_retrieve" in document_result["execution_steps"]
    assert insufficient_result["execution_steps"][-1] == "fallback_answer"
