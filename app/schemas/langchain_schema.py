from __future__ import annotations

from typing import Literal

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


class StrictRequestModel(BaseModel):
    """
    LangChain API 请求基类。

    extra="forbid" 用来拒绝未声明字段，避免调用方传入拼错或多余参数时静默通过。
    """

    model_config = ConfigDict(
        extra="forbid",
    )


class HistoryMessage(StrictRequestModel):
    """
    /langchain/chat 的历史消息结构。
    """

    role: Literal[
        "user",
        "assistant",
    ]

    content: str = Field(
        min_length=1,
        max_length=10000,
    )


class LangChainChatRequest(StrictRequestModel):
    """
    /langchain/chat 请求体。
    """

    query: str = Field(
        min_length=1,
        max_length=3000,
    )

    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=20,
    )


class QueryAnalysisRequest(StrictRequestModel):
    """
    /langchain/analyze-query 请求体。
    """

    query: str = Field(
        min_length=1,
        max_length=1000,
    )


class LangChainAgentRequest(StrictRequestModel):
    """
    /langchain/agent 请求体。
    """

    query: str = Field(
        min_length=1,
        max_length=3000,
    )

    project_id: int = Field(
        default=1,
        ge=1,
    )

    thread_id: str = Field(
        default="default",
        min_length=1,
        max_length=100,
    )

    user_id: str = Field(
        default="local-user",
        min_length=1,
        max_length=100,
    )

    project_root: str = Field(
        default=".",
        min_length=1,
        max_length=1000,
    )

    chunks_path: str = Field(
        default="outputs/chunks.json",
        min_length=1,
        max_length=1000,
    )

    index_path: str = Field(
        default="outputs/vector_index.json",
        min_length=1,
        max_length=1000,
    )

    recursion_limit: int = Field(
        default=20,
        ge=4,
        le=50,
    )

    run_id: str | None = Field(
        default=None,
        max_length=100,
    )

    trace_id: str | None = Field(
        default=None,
        max_length=100,
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
