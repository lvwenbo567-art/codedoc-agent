from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.model_factory import create_chat_model
from langchain_agent.tool_adapter import build_langchain_tools
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from tools.code_doc_tools import build_code_doc_tool_registry
from tools.executor import ToolExecutor


class ToolAgentConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class CodeDocToolAgentDependencies:
    runtime: ToolAgentRuntimeConfig
    model_config: LangChainModelConfig
    model_with_tools: Any#绑定了工具 schema 的模型 model.bind_tools(tools)
    tools: list[BaseTool]
    allowed_tool_names: frozenset[str]#工具白名单


    chat_model: Any | None = None


def build_tool_agent_dependencies(
    *,
    runtime: ToolAgentRuntimeConfig,
    model_config: LangChainModelConfig | None = None,
) -> CodeDocToolAgentDependencies:
    runtime.validate_runtime()#运行前确认 project_root 是合法目录
    effective_model_config = model_config or LangChainModelConfig.from_env()

    if effective_model_config.provider == "mock":
        raise ToolAgentConfigurationError(
            "Day37 Tool Agent 需要支持 Tool Calling 的真实模型；"
            "mock 模式只用于单元测试。"
        )

    registry = build_code_doc_tool_registry(
        project_root=runtime.resolved_project_root,
        chunks_path=runtime.resolved_chunks_path,
        index_path=runtime.resolved_index_path,
        embedding_provider=runtime.embedding_provider,
        embedding_model=runtime.embedding_model,
        embedding_base_url=runtime.embedding_base_url,
        embedding_api_key=runtime.embedding_api_key,
        embedding_timeout_seconds=runtime.embedding_timeout_seconds,
        mock_dimension=runtime.mock_dimension,
        rerank_provider=runtime.rerank_provider,
        rerank_model=runtime.rerank_model,
        rerank_device=runtime.rerank_device,
        rerank_batch_size=runtime.rerank_batch_size,
        rerank_max_length=runtime.rerank_max_length,
        rerank_local_files_only=runtime.rerank_local_files_only,
        query_rewrite_provider=effective_model_config.provider,
        query_rewrite_model=effective_model_config.model_name,
        query_rewrite_base_url=effective_model_config.base_url,
        query_rewrite_api_key=(
            effective_model_config.api_key.get_secret_value()
        ),
        query_rewrite_timeout_seconds=(
            effective_model_config.timeout_seconds
        ),
    )
    executor = ToolExecutor(registry=registry)
    tools = build_langchain_tools(
        registry=registry,
        executor=executor,
    )

    if not tools:
        raise ToolAgentConfigurationError("没有可用的 LangChain Tools")

    model = create_chat_model(effective_model_config)
    model_with_tools = model.bind_tools(tools)#把 tools 的名称、描述、参数 schema 告诉模型。
    allowed_tool_names = frozenset(tool.name for tool in tools)

    return CodeDocToolAgentDependencies(
        runtime=runtime,
        model_config=effective_model_config,
        chat_model=model,
        model_with_tools=model_with_tools,
        tools=tools,
        allowed_tool_names=allowed_tool_names,
    )
