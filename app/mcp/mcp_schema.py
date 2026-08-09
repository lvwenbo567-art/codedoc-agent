from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from config import (
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_DEVICE,
    DEFAULT_RERANK_LOCAL_FILES_ONLY,
    DEFAULT_RERANK_MAX_LENGTH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_PROVIDER,
)


class StrictMcpModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class McpTool(StrictMcpModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    """
    这个工具需要哪些参数？
    参数类型是什么？
    哪些字段必填？
    """


class McpResource(StrictMcpModel):
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"


class McpPrompt(StrictMcpModel):
    name: str
    description: str
    template: str


class McpRuntimeConfig(StrictMcpModel):
    project_root: str = "."
    chunks_path: str = "outputs/chunks.json"
    index_path: str = "outputs/vector_index.json"
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY
    embedding_timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS
    mock_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    rerank_provider: str = DEFAULT_RERANK_PROVIDER
    rerank_model: str = DEFAULT_RERANK_MODEL
    rerank_device: str = DEFAULT_RERANK_DEVICE
    rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE
    rerank_max_length: int = DEFAULT_RERANK_MAX_LENGTH
    rerank_local_files_only: bool = DEFAULT_RERANK_LOCAL_FILES_ONLY


class McpCallToolRequest(McpRuntimeConfig):
    tool_name: str = Field(min_length=1, max_length=80)
    arguments: dict[str, Any] = Field(default_factory=dict)


class McpCallToolResult(StrictMcpModel):
    tool_name: str
    success: bool
    data: Any | None = None
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float = 0.0
