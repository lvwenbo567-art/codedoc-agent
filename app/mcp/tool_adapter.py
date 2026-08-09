from __future__ import annotations

from typing import Any

from mcp.mcp_schema import McpCallToolRequest, McpCallToolResult, McpRuntimeConfig, McpTool
from tools.code_doc_tools import build_code_doc_tool_registry
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry


READ_ONLY_MCP_TOOLS = {
    "get_project_structure",
    "get_symbol_definition",
    "read_file_range",
    "search_code",
    "search_documents",
}


def build_mcp_tool_registry(config: McpRuntimeConfig) -> ToolRegistry:
    return build_code_doc_tool_registry(
        project_root=config.project_root,
        chunks_path=config.chunks_path,
        index_path=config.index_path,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        embedding_base_url=config.embedding_base_url,
        embedding_api_key=config.embedding_api_key,
        embedding_timeout_seconds=config.embedding_timeout_seconds,
        mock_dimension=config.mock_dimension,
        rerank_provider=config.rerank_provider,
        rerank_model=config.rerank_model,
        rerank_device=config.rerank_device,
        rerank_batch_size=config.rerank_batch_size,
        rerank_max_length=config.rerank_max_length,
        rerank_local_files_only=config.rerank_local_files_only,
    )


def _schema_without_title(schema: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(schema)
    cleaned.pop("title", None)
    return cleaned


def list_mcp_tools(config: McpRuntimeConfig) -> list[McpTool]:
    registry = build_mcp_tool_registry(config)
    tools: list[McpTool] = []

    for spec in registry.list_specs():
        if spec.name not in READ_ONLY_MCP_TOOLS:
            continue

        tools.append(
            McpTool(
                name=spec.name,
                description=spec.description,
                input_schema=_schema_without_title(
                    spec.args_model.model_json_schema()
                ),
            )
        )

    return tools


def call_mcp_tool(request: McpCallToolRequest) -> McpCallToolResult:
    if request.tool_name not in READ_ONLY_MCP_TOOLS:
        return McpCallToolResult(
            tool_name=request.tool_name,
            success=False,
            error_code="MCP_TOOL_NOT_ALLOWED",
            error_message=(
                "This MCP endpoint only exposes read-only CodeDoc tools. "
                f"Tool is not allowed: {request.tool_name}"
            ),
        )

    registry = build_mcp_tool_registry(request)
    result = ToolExecutor(registry).execute(
        tool_name=request.tool_name,
        arguments=request.arguments,
    )

    return McpCallToolResult(
        tool_name=result.tool_name,
        success=result.success,
        data=result.data,
        error_code=result.error_code,
        error_message=result.error_message,
        duration_ms=result.duration_ms,
    )
