from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langgraph_agent.routes import (
    route_after_analysis,
    route_after_evidence_assessment,
    route_after_symbol_lookup,
)


def test_route_after_analysis_code_with_symbol():
    assert (
        route_after_analysis(
            {
                "query_type": "code",
                "retrieval_strategy": "original",
                "symbol_name": "keyword_score",
            }
        )
        == "exact_symbol_lookup"
    )


def test_route_after_analysis_code_without_symbol():
    assert route_after_analysis({"query_type": "code"}) == "code_retrieve"


def test_route_after_analysis_document_structure_unknown():
    assert route_after_analysis({"query_type": "document"}) == "document_retrieve"
    assert route_after_analysis({"query_type": "structure"}) == "project_structure"
    assert route_after_analysis({"query_type": "unknown"}) == "fallback_answer"


def test_route_after_symbol_lookup():
    assert (
        route_after_symbol_lookup(
            {
                "evidence": [
                    {
                        "source_path": "app/a.py",
                        "content": "def keyword_score(): pass",
                    }
                ]
            }
        )
        == "assess_evidence"
    )
    assert route_after_symbol_lookup({"evidence": []}) == "code_retrieve"


def test_route_after_evidence_assessment():
    assert (
        route_after_evidence_assessment({"evidence_sufficient": True})
        == "build_answer"
    )
    assert (
        route_after_evidence_assessment({"evidence_sufficient": False})
        == "fallback_answer"
    )
