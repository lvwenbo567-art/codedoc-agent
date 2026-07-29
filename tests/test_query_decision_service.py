from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.decision_schema import QueryDecision
from langgraph_agent.query_decision_service import QueryDecisionService


def build_service() -> QueryDecisionService:
    return QueryDecisionService(
        model_config=LangChainModelConfig(provider="mock")
    )


def test_model_document_decision_is_corrected_for_code_semantic_query():
    class DocumentStructuredModel:
        def invoke(self, messages):
            return QueryDecision(
                query_type="document",
                retrieval_strategy="multi_query",
                symbol_name=None,
                confidence=0.9,
                reason="model thinks this is document",
                decision_method="model",
            )

    service = build_service()
    service._structured_model = DocumentStructuredModel()

    decision = service.analyze(
        "how does multi-query rerank hybrid retrieval pipeline work"
    )

    assert decision.query_type == "code"
    assert decision.retrieval_strategy == "multi_query"
    assert decision.symbol_name is None
    assert decision.decision_method == "model"


def test_model_exception_falls_back_to_rules():
    class BrokenStructuredModel:
        def invoke(self, messages):
            raise RuntimeError("model down")

    service = build_service()
    service._structured_model = BrokenStructuredModel()

    decision = service.analyze("retrieve_with_rerank location")

    assert decision.decision_method == "rule_fallback"
    assert decision.query_type == "code"
    assert decision.retrieval_strategy == "original"


def test_exact_symbol_uses_original_strategy():
    decision = build_service().analyze(
        "retrieve_with_rerank location"
    )

    assert decision.query_type == "code"
    assert decision.retrieval_strategy == "original"
    assert decision.symbol_name == "retrieve_with_rerank"


def test_natural_language_code_uses_multi_query():
    decision = build_service().analyze(
        "how does rerank and hybrid retrieval work"
    )

    assert decision.query_type == "code"
    assert decision.retrieval_strategy == "multi_query"


def test_document_structure_and_unknown_decisions():
    service = build_service()

    assert service.analyze("README startup guide").query_type == "document"
    assert (
        service.analyze("project directories and modules")
        .retrieval_strategy
        == "structure"
    )
    assert service.analyze("what should I eat today").retrieval_strategy == "none"


def test_extract_symbol_candidate_supports_camel_case():
    assert (
        QueryDecisionService.extract_symbol_candidate(
            "RerankClient location"
        )
        == "RerankClient"
    )
