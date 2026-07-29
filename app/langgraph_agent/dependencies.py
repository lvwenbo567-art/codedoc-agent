from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol#Protocol：规定对象必须具备哪些能力.可以理解成一份“接口要求”或者“能力清单”。

from langchain_agent.chat_service import LangChainChatService
from langchain_agent.model_config import LangChainModelConfig
from config import (
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
)
from tools.code_doc_tools import build_code_doc_tool_registry
from tools.executor import ToolExecutor
from langgraph_agent.answer_service import GraphAnswerService
from langgraph_agent.evidence_quality_service import EvidenceQualityService
from langgraph_agent.query_decision_service import QueryDecisionService
from langgraph_agent.rag_runtime import RAGRuntimeConfig


class ToolExecutorProtocol(Protocol):
    def execute(
        self,
        tool_name: str,
        arguments: str | dict[str, Any],
    ) -> Any:
        ...


class ChatServiceProtocol(Protocol):
    def ask(
        self,
        *,
        query: str,
        history: list[Any] | None = None,
    ) -> Any:
        ...



@dataclass(frozen=True)
class CodeDocGraphDependencies:
    """
    LangGraph Node 所需依赖。

    测试时可以替换成 FakeToolExecutor 和 FakeChatService。
    """

    tool_executor: ToolExecutorProtocol
    chat_service: ChatServiceProtocol
    query_decision_service: QueryDecisionService | None = None
    evidence_quality_service: EvidenceQualityService | None = None
    answer_service: GraphAnswerService | None = None
    runtime: RAGRuntimeConfig | None = None


def build_graph_dependencies(
    *,
    runtime: RAGRuntimeConfig | None = None,
    project_root: str | None = None,
    chunks_path: str | None = None,
    index_path: str | None = None,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY,
    embedding_timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    mock_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    model_config: LangChainModelConfig | None = None,
) -> CodeDocGraphDependencies:
    """
    复用当前项目已有 Tool Registry、ToolExecutor 和 LangChainChatService。
    """
    effective_runtime = runtime or RAGRuntimeConfig(
        project_root=project_root or ".",
        chunks_path=chunks_path or "outputs/chunks.json",
        index_path=index_path or "outputs/vector_index.json",
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        embedding_timeout_seconds=embedding_timeout_seconds,
        mock_dimension=mock_dimension,
    )
    effective_runtime.validate()

    registry = build_code_doc_tool_registry(
        project_root=effective_runtime.project_root,
        chunks_path=effective_runtime.chunks_path,
        index_path=effective_runtime.index_path,
        embedding_provider=effective_runtime.embedding_provider,
        embedding_model=effective_runtime.embedding_model,
        embedding_base_url=effective_runtime.embedding_base_url,
        embedding_api_key=effective_runtime.embedding_api_key,
        embedding_timeout_seconds=effective_runtime.embedding_timeout_seconds,
        mock_dimension=effective_runtime.mock_dimension,
        rerank_provider=effective_runtime.rerank_provider,
        rerank_model=effective_runtime.rerank_model,
        rerank_device=effective_runtime.rerank_device,
        rerank_batch_size=effective_runtime.rerank_batch_size,
        rerank_max_length=effective_runtime.rerank_max_length,
        rerank_local_files_only=effective_runtime.rerank_local_files_only,
        query_rewrite_provider=effective_runtime.query_rewrite_provider,
        query_rewrite_model=effective_runtime.query_rewrite_model,
        query_rewrite_base_url=effective_runtime.query_rewrite_base_url,
        query_rewrite_api_key=effective_runtime.query_rewrite_api_key,
        query_rewrite_timeout_seconds=effective_runtime.query_rewrite_timeout_seconds,
    )
    tool_executor = ToolExecutor(registry=registry)
    effective_model_config = model_config or LangChainModelConfig.from_env()
    chat_service = LangChainChatService(
        config=effective_model_config,
    )
    query_decision_service = QueryDecisionService(
        model_config=effective_model_config,
    )
    evidence_quality_service = EvidenceQualityService(
        model_config=effective_model_config,
    )
    answer_service = GraphAnswerService(
        model_config=effective_model_config,
        max_context_chars=effective_runtime.max_context_chars,
    )

    return CodeDocGraphDependencies(
        tool_executor=tool_executor,
        chat_service=chat_service,
        query_decision_service=query_decision_service,
        evidence_quality_service=evidence_quality_service,
        answer_service=answer_service,
        runtime=effective_runtime,
    )
