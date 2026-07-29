from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
        / "app"
    )
)

from langgraph_agent.routes import (
    route_by_evidence_sufficiency,
    route_by_query_type,
)


def test_route_by_query_type():
    assert route_by_query_type({"query_type": "code"}) == "code_search"
    assert route_by_query_type({"query_type": "document"}) == "document_search"
    assert route_by_query_type({"query_type": "structure"}) == "project_structure"
    assert route_by_query_type({"query_type": "unknown"}) == "fallback_answer"


def test_route_by_evidence_sufficiency():
    assert (
        route_by_evidence_sufficiency(
            {"evidence_sufficient": True}
        )
        == "build_answer"
    )
    assert (
        route_by_evidence_sufficiency(
            {"evidence_sufficient": False}
        )
        == "fallback_answer"
    )
