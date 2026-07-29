from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies
from langgraph_agent.tool_agent_nodes import CodeDocToolAgentNodes
from langgraph_agent.tool_agent_state import CodeDocToolAgentState


def build_codedoc_tool_agent_graph(
    dependencies: CodeDocToolAgentDependencies,
    *,
    tool_node: Any | None = None,
) -> Any:
    nodes = CodeDocToolAgentNodes(
        dependencies=dependencies,
        tool_node=tool_node,
    )
    builder = StateGraph(CodeDocToolAgentState)
    '''
    我要创建一个 LangGraph 状态图；
    这个图的共享状态结构是 CodeDocToolAgentState。
    '''
    builder.add_node("initialize", nodes.initialize_node)
    builder.add_node("agent", nodes.call_model_node)
    builder.add_node(
        "controller",
        nodes.controller_node,
        destinations={#因为 Command 是动态路由。LangGraph 需要知道这个节点可能去哪里，才能正确画图和执行。
            "tools": "tools",
            "finalize": "finalize",
            "limit_answer": "limit_answer",
        },
    )
    builder.add_node("tools", nodes.tools_node)
    builder.add_node("finalize", nodes.finalize_node)
    builder.add_node("limit_answer", nodes.limit_answer_node)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "agent")
    builder.add_edge("agent", "controller")
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", END)
    builder.add_edge("limit_answer", END)

    return builder.compile()


def export_tool_agent_mermaid(
    *,
    graph: Any,
    output_path: str,
) -> str:
    mermaid = graph.get_graph().draw_mermaid()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mermaid, encoding="utf-8")

    return mermaid
