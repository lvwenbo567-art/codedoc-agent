from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT / "app"))

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.dependencies import build_graph_dependencies
from langgraph_agent.graph import (
    build_codedoc_agentic_rag_graph,
    export_agentic_rag_mermaid,
)
from langgraph_agent.rag_runtime import RAGRuntimeConfig


def main() -> None:
    runtime = RAGRuntimeConfig(
        project_root=str(PROJECT_ROOT),
        chunks_path=str(PROJECT_ROOT / "outputs" / "chunks.json"),
        index_path=str(PROJECT_ROOT / "outputs" / "vector_index.json"),
    )
    dependencies = build_graph_dependencies(
        runtime=runtime,
        model_config=LangChainModelConfig.from_env(),
    )
    graph = build_codedoc_agentic_rag_graph(dependencies)
    mermaid = export_agentic_rag_mermaid(
        graph=graph,
        output_path=str(
            PROJECT_ROOT / "docs" / "day36_agentic_rag.mmd"
        ),
    )

    print(mermaid)


if __name__ == "__main__":
    main()
