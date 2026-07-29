from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from langgraph_agent.dependencies import CodeDocGraphDependencies
from langgraph_agent.nodes import CodeDocWorkflowNodes
from langgraph_agent.routes import (
    route_after_analysis,
    route_after_evidence_assessment,
    route_after_symbol_lookup,
    route_by_evidence_sufficiency,
    route_by_query_type,
)
from langgraph_agent.state import CodeDocGraphState


def build_codedoc_workflow(
    dependencies: CodeDocGraphDependencies,
) -> Any:
    """
    构建 Day35 确定性 CodeDoc Workflow。
    """
    nodes = CodeDocWorkflowNodes(dependencies=dependencies)
    builder = StateGraph(CodeDocGraphState)

    builder.add_node("initialize", nodes.initialize_node)
    builder.add_node("classify_query", nodes.classify_query_node)
    builder.add_node("code_search", nodes.code_search_node)
    builder.add_node("document_search", nodes.document_search_node)
    builder.add_node("project_structure", nodes.project_structure_node)
    builder.add_node("check_evidence", nodes.check_evidence_node)
    builder.add_node("build_answer", nodes.build_answer_node)
    builder.add_node("fallback_answer", nodes.fallback_answer_node)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "classify_query")
    builder.add_conditional_edges(
        "classify_query",
        route_by_query_type,
        {
            "code_search": "code_search",
            "document_search": "document_search",
            "project_structure": "project_structure",
            "fallback_answer": "fallback_answer",
        },
    )
    builder.add_edge("code_search", "check_evidence")
    builder.add_edge("document_search", "check_evidence")
    builder.add_edge("project_structure", "check_evidence")
    builder.add_conditional_edges(
        "check_evidence",
        route_by_evidence_sufficiency,
        {
            "build_answer": "build_answer",
            "fallback_answer": "fallback_answer",
        },
    )
    builder.add_edge("build_answer", END)
    builder.add_edge("fallback_answer", END)

    return builder.compile()


def build_codedoc_agentic_rag_graph(
    dependencies: CodeDocGraphDependencies,
) -> Any:
    """
    构建 Day36 Agentic RAG Workflow v1。
    """
    nodes = CodeDocWorkflowNodes(dependencies=dependencies)
    builder = StateGraph(CodeDocGraphState)

    builder.add_node("initialize", nodes.initialize_node)
    builder.add_node("analyze_query", nodes.analyze_query_node)
    builder.add_node("exact_symbol_lookup", nodes.exact_symbol_lookup_node)
    builder.add_node("code_retrieve", nodes.code_retrieve_node)
    builder.add_node("document_retrieve", nodes.document_retrieve_node)
    builder.add_node("project_structure", nodes.project_structure_node)
    builder.add_node("assess_evidence", nodes.assess_evidence_node)
    builder.add_node("build_answer", nodes.build_answer_node)
    builder.add_node("fallback_answer", nodes.fallback_answer_node)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "analyze_query")
    builder.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "exact_symbol_lookup": "exact_symbol_lookup",
            "code_retrieve": "code_retrieve",
            "document_retrieve": "document_retrieve",
            "project_structure": "project_structure",
            "fallback_answer": "fallback_answer",
        },
    )
    builder.add_conditional_edges(
        "exact_symbol_lookup",
        route_after_symbol_lookup,
        {
            "assess_evidence": "assess_evidence",
            "code_retrieve": "code_retrieve",
        },
    )
    builder.add_edge("code_retrieve", "assess_evidence")
    builder.add_edge("document_retrieve", "assess_evidence")
    builder.add_edge("project_structure", "assess_evidence")
    builder.add_conditional_edges(
        "assess_evidence",
        route_after_evidence_assessment,
        {
            "build_answer": "build_answer",
            "fallback_answer": "fallback_answer",
        },
    )
    builder.add_edge("build_answer", END)
    builder.add_edge("fallback_answer", END)

    return builder.compile()


def export_workflow_mermaid(
    *,
    graph: Any,
    output_path: str,
) -> str:
    mermaid = graph.get_graph().draw_mermaid()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mermaid, encoding="utf-8")

    return mermaid


def export_agentic_rag_mermaid(
    *,
    graph: Any,
    output_path: str,
) -> str:
    return export_workflow_mermaid(
        graph=graph,
        output_path=output_path,
    )
