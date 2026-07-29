from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langgraph_agent.decision_schema import EvidenceAssessment, QueryDecision
from langgraph_agent.dependencies import CodeDocGraphDependencies
from langgraph_agent.nodes import CodeDocWorkflowNodes
from langgraph_agent.rag_runtime import RAGRuntimeConfig


@dataclass
class FakeToolResult:
    success: bool
    data: Any = None
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float = 0


class FakeToolExecutor:
    def __init__(
        self,
        symbol_hit: bool = True,
        degraded: bool = False,
    ) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.symbol_hit = symbol_hit
        self.degraded = degraded

    def execute(self, tool_name: str, arguments: dict | str) -> FakeToolResult:
        assert isinstance(arguments, dict)
        self.calls.append((tool_name, arguments))

        if tool_name == "get_symbol_definition":
            return FakeToolResult(
                success=True,
                data={
                    "result_count": 1 if self.symbol_hit else 0,
                    "results": [
                        {
                            "chunk_id": "symbol-1",
                            "source_path": "app/a.py",
                            "symbol_name": "keyword_score",
                            "content": "def keyword_score(): pass",
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
                    "degraded": self.degraded,
                    "degrade_reason": (
                        "Rerank 服务不可用，已降级到 Hybrid Search"
                        if self.degraded
                        else None
                    ),
                    "results": [
                        {
                            "chunk_id": "chunk-1",
                            "source_path": "app/a.py",
                            "chunk_type": "code",
                            "content": "def a(): pass",
                            "score": 0.8,
                        }
                    ],
                },
            )

        if tool_name == "get_project_structure":
            return FakeToolResult(
                success=True,
                data={
                    "entries": [
                        {"path": "app", "type": "directory"},
                        {"path": "app/a.py", "type": "file"},
                    ]
                },
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
            coverage_score=0.8 if self.sufficient else 0.1,
            reason="ok" if self.sufficient else "bad",
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


def build_nodes(
    *,
    decision: QueryDecision | None = None,
    symbol_hit: bool = True,
    sufficient: bool = True,
    degraded: bool = False,
) -> tuple[CodeDocWorkflowNodes, FakeToolExecutor]:
    executor = FakeToolExecutor(
        symbol_hit=symbol_hit,
        degraded=degraded,
    )
    dependencies = CodeDocGraphDependencies(
        tool_executor=executor,
        chat_service=FakeChatService(),
        query_decision_service=FakeDecisionService(
            decision
            or QueryDecision(
                query_type="code",
                retrieval_strategy="original",
                symbol_name="keyword_score",
                confidence=0.9,
                reason="test",
                decision_method="rule",
            )
        ),
        evidence_quality_service=FakeEvidenceQualityService(sufficient=sufficient),
        answer_service=FakeAnswerService(),
        runtime=RAGRuntimeConfig(candidate_top_k=20, final_top_k=5),
    )
    return CodeDocWorkflowNodes(dependencies=dependencies), executor


def test_analyze_query_writes_decision_to_state():
    nodes, _ = build_nodes()

    result = nodes.analyze_query_node({"query": "keyword_score 在哪里？"})

    assert result["query_type"] == "code"
    assert result["retrieval_strategy"] == "original"
    assert result["symbol_name"] == "keyword_score"


def test_exact_symbol_lookup_hit_and_miss():
    nodes, _ = build_nodes(symbol_hit=True)
    hit = nodes.exact_symbol_lookup_node({"symbol_name": "keyword_score"})

    assert hit["retrieval_metadata"]["symbol_lookup_hit"] is True
    assert hit["evidence"][0]["symbol_name"] == "keyword_score"

    nodes, _ = build_nodes(symbol_hit=False)
    miss = nodes.exact_symbol_lookup_node({"symbol_name": "keyword_score"})

    assert miss["retrieval_metadata"]["symbol_lookup_hit"] is False


def test_code_and_document_retrieve_pass_chunk_strategy():
    nodes, executor = build_nodes()

    code = nodes.code_retrieve_node(
        {
            "query": "自然语言代码问题",
            "retrieval_strategy": "multi_query",
        }
    )
    document = nodes.document_retrieve_node(
        {
            "query": "README 启动方式",
            "retrieval_strategy": "multi_query",
        }
    )

    assert executor.calls[0][0] == "search_code"
    assert executor.calls[0][1]["query_strategy"] == "multi_query"
    assert code["retrieval_metadata"]["query_strategy"] == "multi_query"
    assert executor.calls[1][0] == "search_documents"
    assert document["evidence"]


def test_rerank_degradation_is_written_to_state():
    nodes, _ = build_nodes(degraded=True)

    result = nodes.code_retrieve_node(
        {
            "query": "多路检索和重排",
            "retrieval_strategy": "multi_query",
        }
    )

    assert result["degraded"] is True
    assert result["degrade_reasons"] == [
        "Rerank 服务不可用，已降级到 Hybrid Search"
    ]


def test_assess_evidence_and_build_answer_nodes():
    nodes, _ = build_nodes()
    evidence = [
        {
            "source_path": "app/a.py",
            "evidence_type": "code_search",
            "content": "def a(): pass",
        }
    ]

    assessed = nodes.assess_evidence_node(
        {
            "query": "a 是什么？",
            "query_type": "code",
            "evidence": evidence,
        }
    )
    answered = nodes.build_answer_node(
        {
            "query": "a 是什么？",
            "evidence": evidence,
        }
    )

    assert assessed["evidence_sufficient"] is True
    assert answered["answer"] == "answer [Source 1]"
    assert answered["citations"][0]["citation_id"] == "Source 1"
