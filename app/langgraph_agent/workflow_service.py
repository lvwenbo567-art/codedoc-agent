from __future__ import annotations

import asyncio
from typing import Any

from config import (
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
)
from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.dependencies import (
    CodeDocGraphDependencies,
    build_graph_dependencies,
)
from langgraph_agent.graph import (
    build_codedoc_agentic_rag_graph,
    build_codedoc_workflow,
)
from langgraph_agent.rag_runtime import RAGRuntimeConfig
from langgraph_agent.state import CodeDocGraphState


class CodeDocWorkflowExecutionError(RuntimeError):
    pass


class CodeDocWorkflowService:
    """
    Day35 确定性 Workflow 服务入口。
    """

    def __init__(
        self,
        *,
        dependencies: CodeDocGraphDependencies | None = None,
        project_root: str = ".",
        chunks_path: str = "outputs/chunks.json",
        index_path: str = "outputs/vector_index.json",
        embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
        embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY,
        embedding_timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
        mock_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        model_config: LangChainModelConfig | None = None,
        graph: Any | None = None,
    ) -> None:
        self.dependencies = dependencies or build_graph_dependencies(
            project_root=project_root,
            chunks_path=chunks_path,
            index_path=index_path,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_timeout_seconds=embedding_timeout_seconds,
            mock_dimension=mock_dimension,
            model_config=model_config,
        )
        self.graph = graph or build_codedoc_workflow(self.dependencies)

    @staticmethod
    def _build_initial_state(
        *,
        query: str,
        project_id: int,
    ) -> CodeDocGraphState:
        query = query.strip()

        if not query:
            raise ValueError("query 不能为空")

        return {
            "query": query,
            "project_id": project_id,
            "evidence": [],
            "execution_steps": [],
        }

    def run(
        self,
        *,
        query: str,
        project_id: int = 1,
    ) -> CodeDocGraphState:
        return self.graph.invoke(
            self._build_initial_state(
                query=query,
                project_id=project_id,
            )
        )

    async def arun(
        self,
        *,
        query: str,
        project_id: int = 1,
    ) -> CodeDocGraphState:
        initial_state = self._build_initial_state(
            query=query,
            project_id=project_id,
        )

        if hasattr(self.graph, "ainvoke"):
            return await self.graph.ainvoke(initial_state)

        return await asyncio.to_thread(
            self.graph.invoke,
            initial_state,
        )


class CodeDocAgenticRAGService:
    """
    Day36 Agentic RAG Workflow v1 服务入口。
    """

    def __init__(
        self,
        *,
        dependencies: CodeDocGraphDependencies | None = None,
        runtime: RAGRuntimeConfig | None = None,
        model_config: LangChainModelConfig | None = None,
        graph: Any | None = None,
    ) -> None:
        self.runtime = runtime or RAGRuntimeConfig()
        self.runtime.validate()
        self.dependencies = dependencies or build_graph_dependencies(
            runtime=self.runtime,
            model_config=model_config,
        )
        self.graph = graph or build_codedoc_agentic_rag_graph(
            self.dependencies
        )

    @staticmethod
    def _build_initial_state(
        *,
        query: str,
        project_id: int,
    ) -> CodeDocGraphState:
        query = query.strip()

        if not query:
            raise ValueError("query 不能为空")

        return {
            "query": query,
            "project_id": project_id,
            "evidence": [],
            "execution_steps": [],
            "degrade_reasons": [],
        }

    def run(
        self,
        *,
        query: str,
        project_id: int = 1,
        recursion_limit: int = 20,
    ) -> CodeDocGraphState:
        try:
            return self.graph.invoke(
                self._build_initial_state(
                    query=query,
                    project_id=project_id,
                ),
                config={"recursion_limit": recursion_limit},
            )
        except Exception as exc:
            raise CodeDocWorkflowExecutionError(str(exc)) from exc

    async def arun(
        self,
        *,
        query: str,
        project_id: int = 1,
        recursion_limit: int = 20,
    ) -> CodeDocGraphState:
        initial_state = self._build_initial_state(
            query=query,
            project_id=project_id,
        )
        config = {"recursion_limit": recursion_limit}

        try:
            if hasattr(self.graph, "ainvoke"):
                return await self.graph.ainvoke(
                    initial_state,
                    config=config,
                )

            return await asyncio.to_thread(
                self.graph.invoke,
                initial_state,
                config,
            )
        except Exception as exc:
            raise CodeDocWorkflowExecutionError(str(exc)) from exc
