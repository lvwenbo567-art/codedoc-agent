from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VectorStoreBackend = Literal["json", "qdrant"]


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


class VectorStoreConfig(BaseModel):
    """VectorStore 后端配置，支持从环境变量切换。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: VectorStoreBackend = "json"
    json_index_path: str = Field(default="outputs/vector_index.json", min_length=1)
    project_id: int = Field(default=1, gt=0)
    qdrant_url: str = Field(default="http://localhost:6333", min_length=1)
    qdrant_api_key: str | None = None
    qdrant_collection: str = Field(default="codedoc_chunks_v1", min_length=1, max_length=255)
    qdrant_prefer_grpc: bool = False
    qdrant_timeout_seconds: float = Field(default=10, gt=0, le=120)

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        backend = os.getenv("VECTOR_STORE_BACKEND", "json").strip().lower()

        if backend not in {"json", "qdrant"}:
            raise ValueError("VECTOR_STORE_BACKEND 必须是 json 或 qdrant")

        api_key = os.getenv("QDRANT_API_KEY")

        if api_key is not None:
            api_key = api_key.strip() or None

        return cls(
            backend=backend,
            json_index_path=os.getenv("VECTOR_INDEX_PATH", os.getenv("CODEDOC_VECTOR_INDEX_PATH", "outputs/vector_index.json")),
            project_id=int(os.getenv("VECTOR_STORE_PROJECT_ID", "1")),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=api_key,
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "codedoc_chunks_v1"),
            qdrant_prefer_grpc=_read_bool("QDRANT_PREFER_GRPC", False),
            qdrant_timeout_seconds=float(os.getenv("QDRANT_TIMEOUT_SECONDS", "10")),
        )
