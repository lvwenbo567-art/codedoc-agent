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
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_RAG_TOP_K,
    DEFAULT_VECTOR_INDEX_PATH,
)


class ScanRequest(BaseModel):
    """
    /scan request body.
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
    /search request body.
    """

    chunks_path: str = "outputs/chunks.json"
    query: str
    top_k: int = 5


class EvalRequest(BaseModel):
    """
    /eval request body.
    """

    chunks_path: str = "outputs/chunks.json"
    eval_path: str = "data/eval_queries.json"
    top_k: int = 5


class IndexRequest(BaseModel):
    """
    /index request body.
    """

    chunks_path: str = "outputs/chunks.json"
    output_path: str = DEFAULT_VECTOR_INDEX_PATH
    model_name: str = DEFAULT_EMBEDDING_MODEL
    dimension: int = DEFAULT_EMBEDDING_DIMENSION


class VectorSearchRequest(BaseModel):
    """
    /vector_search request body.
    """

    index_path: str = DEFAULT_VECTOR_INDEX_PATH
    query: str
    top_k: int = 5
    model_name: str = DEFAULT_EMBEDDING_MODEL
    dimension: int = DEFAULT_EMBEDDING_DIMENSION
    chunk_type: str | None = None


class AskRequest(BaseModel):
    """
    /ask request body.
    """

    query: str
    index_path: str = DEFAULT_VECTOR_INDEX_PATH
    top_k: int = Field(default=DEFAULT_RAG_TOP_K, gt=0)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    dimension: int = Field(default=DEFAULT_EMBEDDING_DIMENSION, gt=0)
    chat_provider: Literal["mock", "openai_compatible"] = DEFAULT_CHAT_PROVIDER
    chat_model: str = DEFAULT_CHAT_MODEL
    chat_base_url: str = DEFAULT_CHAT_BASE_URL
    chat_api_key: str = DEFAULT_CHAT_API_KEY
    chat_timeout_seconds: float = Field(default=DEFAULT_CHAT_TIMEOUT_SECONDS, gt=0)
    temperature: float = Field(default=DEFAULT_CHAT_TEMPERATURE, ge=0, le=2)
    max_tokens: int = Field(default=DEFAULT_CHAT_MAX_TOKENS, gt=0)
    chunk_type: str | None = None
    max_context_chars: int = Field(default=DEFAULT_MAX_CONTEXT_CHARS, gt=0)
