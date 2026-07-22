from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from config import (
    DEFAULT_CHAT_API_KEY,
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_CHAT_MAX_TOKENS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_TIMEOUT_SECONDS,
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
    DEFAULT_VECTOR_INDEX_PATH,
)


class FunctionCallRequest(BaseModel):
    """
    /agent/function-call 接口请求体。

    注意：
    - query、模型配置和检索配置由 API 用户传入；
    - 具体工具参数仍由模型生成；
    - project_root、chunks_path、index_path 会由服务端绑定到 Tool Registry，
      不作为模型可自由生成的工具参数。
    """

    query: str = Field(
        min_length=1,
        max_length=1000,
    )
    project_root: str = "."
    chunks_path: str = "outputs/chunks.json"
    index_path: str = DEFAULT_VECTOR_INDEX_PATH
    max_steps: int = Field(default=4, ge=1, le=10)
    provider: Literal["mock", "openai_compatible"] = DEFAULT_CHAT_PROVIDER
    model_name: str = DEFAULT_CHAT_MODEL
    base_url: str = DEFAULT_CHAT_BASE_URL
    api_key: str = DEFAULT_CHAT_API_KEY
    timeout_seconds: float = Field(
        default=DEFAULT_CHAT_TIMEOUT_SECONDS,
        gt=0,
    )
    temperature: float = Field(
        default=DEFAULT_CHAT_TEMPERATURE,
        ge=0,
        le=2,
    )
    max_tokens: int = Field(
        default=DEFAULT_CHAT_MAX_TOKENS,
        gt=0,
    )
    embedding_provider: Literal["mock", "ollama", "openai_compatible"] = (
        DEFAULT_EMBEDDING_PROVIDER
    )
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY
    embedding_timeout_seconds: float = Field(
        default=DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
        gt=0,
    )
    mock_dimension: int = Field(
        default=DEFAULT_EMBEDDING_DIMENSION,
        gt=0,
    )
    dimension: int | None = Field(
        default=None,
        gt=0,
    )
    rerank_provider: Literal["mock", "sentence_transformers"] = (
        DEFAULT_RERANK_PROVIDER
    )
    rerank_model: str = DEFAULT_RERANK_MODEL
    rerank_device: str = DEFAULT_RERANK_DEVICE
    rerank_batch_size: int = Field(
        default=DEFAULT_RERANK_BATCH_SIZE,
        gt=0,
    )
    rerank_max_length: int = Field(
        default=DEFAULT_RERANK_MAX_LENGTH,
        gt=0,
    )
    rerank_local_files_only: bool = DEFAULT_RERANK_LOCAL_FILES_ONLY
