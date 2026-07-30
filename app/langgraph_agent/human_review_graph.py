from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from langgraph_agent.human_review_nodes import HumanReviewToolAgentNodes
from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies
from langgraph_agent.tool_agent_state import CodeDocToolAgentState


def build_human_review_tool_agent_graph(
    dependencies: CodeDocToolAgentDependencies,
    *,
    tool_node: Any | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """
    构建支持 interrupt / approve / reject / edit 的 Tool Agent Graph。
    """
    nodes = HumanReviewToolAgentNodes(
        dependencies=dependencies,
        tool_node=tool_node,
    )
    builder = StateGraph(CodeDocToolAgentState)

    builder.add_node("initialize", nodes.initialize_node)
    builder.add_node("agent", nodes.call_model_node)
    builder.add_node(
        "controller",
        nodes.controller_node,
        destinations={
            "human_review": "human_review",
            "prepare_tools": "prepare_tools",
            "finalize": "finalize",
            "limit_answer": "limit_answer",
        },
    )
    builder.add_node(
        "human_review",
        nodes.human_review_node,
        destinations={
            "prepare_tools": "prepare_tools",
            "agent": "agent",
            "limit_answer": "limit_answer",
        },
    )
    builder.add_node(
        "prepare_tools",
        nodes.prepare_tools_node,
        destinations={
            "tools": "tools",
            "limit_answer": "limit_answer",
        },
    )
    builder.add_node("tools", nodes.execute_tools_node)
    builder.add_node("finalize", nodes.finalize_node)
    builder.add_node("limit_answer", nodes.limit_answer_node)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "agent")
    builder.add_edge("agent", "controller")
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", END)
    builder.add_edge("limit_answer", END)

    return builder.compile(checkpointer=checkpointer)
