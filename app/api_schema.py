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
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DB_PATH,
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_RAG_TOP_K,
    DEFAULT_VECTOR_INDEX_PATH,
)


EmbeddingProvider = Literal[
    "mock",
    "ollama",
    "openai_compatible",
]


class ScanRequest(BaseModel):
    """
    /scan 接口请求体，用于扫描项目并生成 chunks。
    """

    project_path: str
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_CHUNK_OVERLAP
    save_chunks: bool = False
    output_path: str = "outputs/chunks.json"
    save_to_db: bool = True
    db_path: str = DEFAULT_DB_PATH


class SearchRequest(BaseModel):
    """
    /search 接口请求体，用于从 chunks.json 中做关键词检索。
    """

    chunks_path: str = "outputs/chunks.json"
    query: str
    top_k: int = 5


class EvalRequest(BaseModel):
    """
    /eval 接口请求体，用于执行检索评估。
    """

    chunks_path: str = "outputs/chunks.json"
    eval_path: str = "data/eval_queries.json"
    top_k: int = 5


class IndexRequest(BaseModel):
    """
    /index 接口请求体，用于构建向量索引。
    """

    chunks_path: str = "outputs/chunks.json"
    output_path: str = DEFAULT_VECTOR_INDEX_PATH
    embedding_provider: EmbeddingProvider = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # Backward-compatible aliases used by Day19/20 tests and old API examples.
    model_name: str | None = None
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY
    embedding_timeout_seconds: float = Field(
        default=DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
        gt=0,
    )
    mock_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    dimension: int | None = None
    batch_size: int = Field(default=DEFAULT_EMBEDDING_BATCH_SIZE, gt=0)


class VectorSearchRequest(BaseModel):
    """
    /vector_search 接口请求体，用于执行向量检索。
    """

    index_path: str = DEFAULT_VECTOR_INDEX_PATH
    query: str
    top_k: int = 5
    embedding_provider: EmbeddingProvider = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # Backward-compatible aliases used by Day19/20 tests and old API examples.
    model_name: str | None = None
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY
    embedding_timeout_seconds: float = Field(
        default=DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
        gt=0,
    )
    mock_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    dimension: int | None = None
    chunk_type: str | None = None


class AskRequest(BaseModel):
    """
    /ask 接口请求体，用于执行 RAG 问答。
    """

    query: str
    index_path: str = DEFAULT_VECTOR_INDEX_PATH
    top_k: int = Field(default=DEFAULT_RAG_TOP_K, gt=0)
    embedding_provider: EmbeddingProvider = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # Backward-compatible alias used by old API examples.
    model_name: str | None = None
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY
    embedding_timeout_seconds: float = Field(
        default=DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
        gt=0,
    )
    mock_dimension: int = Field(default=DEFAULT_EMBEDDING_DIMENSION, gt=0)
    dimension: int | None = Field(default=None, gt=0)
    chat_provider: Literal["mock", "openai_compatible"] = DEFAULT_CHAT_PROVIDER
    chat_model: str = DEFAULT_CHAT_MODEL
    chat_base_url: str = DEFAULT_CHAT_BASE_URL
    chat_api_key: str = DEFAULT_CHAT_API_KEY
    chat_timeout_seconds: float = Field(default=DEFAULT_CHAT_TIMEOUT_SECONDS, gt=0)
    temperature: float = Field(default=DEFAULT_CHAT_TEMPERATURE, ge=0, le=2)
    max_tokens: int = Field(default=DEFAULT_CHAT_MAX_TOKENS, gt=0)
    chunk_type: str | None = None
    max_context_chars: int = Field(default=DEFAULT_MAX_CONTEXT_CHARS, gt=0)
