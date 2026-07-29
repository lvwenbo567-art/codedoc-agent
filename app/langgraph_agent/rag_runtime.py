from __future__ import annotations

from dataclasses import dataclass

from config import (
    DEFAULT_CHAT_API_KEY,
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_CHAT_TIMEOUT_SECONDS,
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_CANDIDATE_TOP_K,
    DEFAULT_RERANK_DEVICE,
    DEFAULT_RERANK_FINAL_TOP_K,
    DEFAULT_RERANK_LOCAL_FILES_ONLY,
    DEFAULT_RERANK_MAX_LENGTH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_PROVIDER,
)


@dataclass(frozen=True)
class RAGRuntimeConfig:
    project_root: str = "."
    chunks_path: str = "outputs/chunks.json"
    index_path: str = "outputs/vector_index.json"
    candidate_top_k: int = DEFAULT_RERANK_CANDIDATE_TOP_K
    final_top_k: int = DEFAULT_RERANK_FINAL_TOP_K
    rewrite_count: int = 2
    keyword_weight: float = 0.4
    vector_weight: float = 0.6
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
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
    query_rewrite_provider: str = DEFAULT_CHAT_PROVIDER
    query_rewrite_model: str = DEFAULT_CHAT_MODEL
    query_rewrite_base_url: str = DEFAULT_CHAT_BASE_URL
    query_rewrite_api_key: str = DEFAULT_CHAT_API_KEY
    query_rewrite_timeout_seconds: float = DEFAULT_CHAT_TIMEOUT_SECONDS

    def validate(self) -> None:
        if self.candidate_top_k <= 0:
            raise ValueError("candidate_top_k 必须大于 0")

        if self.final_top_k <= 0:
            raise ValueError("final_top_k 必须大于 0")

        if self.final_top_k > self.candidate_top_k:
            raise ValueError("final_top_k 不能大于 candidate_top_k")

        if self.rewrite_count <= 0:
            raise ValueError("rewrite_count 必须大于 0")

        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars 必须大于 0")
