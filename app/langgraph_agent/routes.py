from __future__ import annotations

from typing import Literal

from langgraph_agent.state import CodeDocGraphState


AnalysisRoute = Literal[
    "exact_symbol_lookup",
    "code_retrieve",
    "document_retrieve",
    "project_structure",
    "fallback_answer",
]


SymbolRoute = Literal[
    "assess_evidence",
    "code_retrieve",
]


EvidenceRoute = Literal[
    "build_answer",
    "fallback_answer",
]


def route_by_query_type(
    state: CodeDocGraphState,
) -> str:
    """
    根据 query_type 决定进入哪个检索节点。
    """
    query_type = state.get(
        "query_type",
        "unknown",
    )

    if query_type == "code":
        return "code_search"

    if query_type == "document":
        return "document_search"

    if query_type == "structure":
        return "project_structure"

    return "fallback_answer"


def route_by_evidence_sufficiency(
    state: CodeDocGraphState,
) -> str:
    """
    根据证据是否充分决定生成答案或降级回答。
    """
    if state.get("evidence_sufficient") is True:
        return "build_answer"

    return "fallback_answer"


def route_after_analysis(
    state: CodeDocGraphState,
) -> AnalysisRoute:
    query_type = state.get(
        "query_type",
        "unknown",
    )

    symbol_name = state.get(
        "symbol_name"
    )

    if (
        query_type == "code"
        and symbol_name
    ):
        return "exact_symbol_lookup"

    if query_type == "code":
        return "code_retrieve"

    if query_type == "document":
        return "document_retrieve"

    if query_type == "structure":
        return "project_structure"

    return "fallback_answer"


def route_after_symbol_lookup(
    state: CodeDocGraphState,
) -> SymbolRoute:
    evidence = state.get(
        "evidence",
        [],
    )

    if evidence:
        return "assess_evidence"

    return "code_retrieve"


def route_after_evidence_assessment(
    state: CodeDocGraphState,
) -> EvidenceRoute:
    if state.get(
        "evidence_sufficient",
        False,
    ):
        return "build_answer"

    return "fallback_answer"
