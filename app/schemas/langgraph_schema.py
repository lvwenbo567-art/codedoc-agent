from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import (
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


class StrictLangGraphModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LangGraphWorkflowRequest(StrictLangGraphModel):
    query: str = Field(min_length=1, max_length=3000)
    project_id: int = Field(default=1, ge=1)
    project_root: str = Field(default=".", min_length=1, max_length=1000)
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
    candidate_top_k: int = Field(
        default=DEFAULT_RERANK_CANDIDATE_TOP_K,
        ge=1,
        le=50,
    )
    final_top_k: int = Field(
        default=DEFAULT_RERANK_FINAL_TOP_K,
        ge=1,
        le=10,
    )
    rewrite_count: int = Field(default=2, ge=1, le=5)
    keyword_weight: float = Field(default=0.4, ge=0, le=1)
    vector_weight: float = Field(default=0.6, ge=0, le=1)
    max_context_chars: int = Field(
        default=DEFAULT_MAX_CONTEXT_CHARS,
        gt=0,
        le=30000,
    )
    recursion_limit: int = Field(default=20, ge=5, le=100)
    embedding_provider: Literal["mock", "ollama", "openai_compatible"] = (
        DEFAULT_EMBEDDING_PROVIDER
    )
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        min_length=1,
        max_length=200,
    )
    embedding_base_url: str = Field(
        default=DEFAULT_EMBEDDING_BASE_URL,
        max_length=1000,
    )
    embedding_api_key: str = Field(
        default=DEFAULT_EMBEDDING_API_KEY,
        max_length=1000,
    )
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

    @model_validator(mode="after")
    def validate_top_k(self) -> "LangGraphWorkflowRequest":
        if self.final_top_k > self.candidate_top_k:
            raise ValueError("final_top_k 不能大于 candidate_top_k")

        return self


class LangGraphCitation(StrictLangGraphModel):
    citation_id: str
    source_path: str
    chunk_id: str | None = None
    score: float | None = None
    start_line: int | None = None
    end_line: int | None = None


class LangGraphWorkflowResponse(StrictLangGraphModel):
    query: str
    project_id: int
    query_type: Literal[
        "code",
        "document",
        "structure",
        "unknown",
    ]
    answer: str
    retrieval_strategy: Literal[
        "original",
        "multi_query",
        "structure",
        "none",
    ] = "none"
    symbol_name: str | None = None
    query_decision: dict[str, Any] = Field(default_factory=dict)
    evidence_assessment: dict[str, Any] = Field(default_factory=dict)
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    answer_quality: dict[str, Any] = Field(default_factory=dict)
    degraded: bool = False
    degrade_reasons: list[str] = Field(default_factory=list)
    execution_steps: list[str] = Field(default_factory=list)
    evidence_sufficient: bool = False
    error_message: str | None = None
