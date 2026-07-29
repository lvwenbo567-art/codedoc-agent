from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT / "app"))

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.dependencies import build_graph_dependencies
from langgraph_agent.graph import (
    build_codedoc_workflow,
    export_workflow_mermaid,
)


def main() -> None:
    dependencies = build_graph_dependencies(
        project_root=str(PROJECT_ROOT),
        chunks_path=str(PROJECT_ROOT / "outputs" / "chunks.json"),
        index_path=str(PROJECT_ROOT / "outputs" / "vector_index.json"),
        model_config=LangChainModelConfig.from_env(),
    )
    graph = build_codedoc_workflow(dependencies)
    mermaid = export_workflow_mermaid(
        graph=graph,
        output_path=str(
            PROJECT_ROOT
            / "docs"
            / "day35_codedoc_workflow.mmd"
        ),
    )

    print(mermaid)


if __name__ == "__main__":
    main()
