from __future__ import annotations

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

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


class ToolAgentRuntimeConfig(BaseModel):
    """
    Day37 Tool Agent 运行配置。

    project_root、chunks_path、index_path 由后端绑定给工具使用，
    不暴露为模型可随意生成的 Tool 参数。
    """

    model_config = ConfigDict(extra="forbid")

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
    max_model_calls: int = Field(default=6, ge=1, le=20)
    max_tool_calls: int = Field(default=10, ge=1, le=50)
    max_identical_tool_calls: int = Field(default=2, ge=1, le=5)
    max_model_messages: int = Field(default=18, ge=4, le=100)
    trace_content_chars: int = Field(default=3000, ge=500, le=12000)
    embedding_provider: str = Field(default=DEFAULT_EMBEDDING_PROVIDER)
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
    mock_dimension: int = Field(default=DEFAULT_EMBEDDING_DIMENSION, gt=0)
    rerank_provider: str = Field(default=DEFAULT_RERANK_PROVIDER)
    rerank_model: str = Field(default=DEFAULT_RERANK_MODEL, max_length=1000)
    rerank_device: str = Field(default=DEFAULT_RERANK_DEVICE, max_length=100)
    rerank_batch_size: int = Field(default=DEFAULT_RERANK_BATCH_SIZE, gt=0)
    rerank_max_length: int = Field(default=DEFAULT_RERANK_MAX_LENGTH, gt=0)
    rerank_local_files_only: bool = DEFAULT_RERANK_LOCAL_FILES_ONLY
    enable_human_review: bool = True
    approval_required_tools: tuple[str, ...] = (
        "run_project_tests",
    )

    @field_validator("approval_required_tools")
    @classmethod
    def validate_approval_tools(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized: list[str] = []

        for tool_name in value:
            name = tool_name.strip()

            if not name:
                raise ValueError(
                    "approval_required_tools 不能包含空工具名"
                )

            if name not in normalized:
                normalized.append(name)

        return tuple(normalized)

    @model_validator(mode="after")
    def validate_runtime(self) -> "ToolAgentRuntimeConfig":
        root = Path(self.project_root).resolve()

        if not root.exists():
            raise ValueError(f"project_root 不存在：{root}")

        if not root.is_dir():
            raise ValueError(f"project_root 不是目录：{root}")

        return self

    @property
    def resolved_project_root(self) -> str:
        return str(Path(self.project_root).resolve())

    @property
    def resolved_chunks_path(self) -> str:
        return str(Path(self.chunks_path).resolve())

    @property
    def resolved_index_path(self) -> str:
        return str(Path(self.index_path).resolve())
