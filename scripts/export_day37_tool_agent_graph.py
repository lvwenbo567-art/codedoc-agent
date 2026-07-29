from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT / "app"))

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import build_tool_agent_dependencies
from langgraph_agent.tool_agent_graph import (
    build_codedoc_tool_agent_graph,
    export_tool_agent_mermaid,
)


runtime = ToolAgentRuntimeConfig(
    project_root=str(PROJECT_ROOT),
    chunks_path=str(PROJECT_ROOT / "outputs" / "chunks.json"),
    index_path=str(PROJECT_ROOT / "outputs" / "vector_index.json"),
)

dependencies = build_tool_agent_dependencies(
    runtime=runtime,
    model_config=LangChainModelConfig.from_env(),
)

graph = build_codedoc_tool_agent_graph(dependencies)

print(
    export_tool_agent_mermaid(
        graph=graph,
        output_path=str(PROJECT_ROOT / "docs" / "day37_tool_agent.mmd"),
    )
)
