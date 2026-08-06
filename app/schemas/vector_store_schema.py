from __future__ import annotations

from pydantic import BaseModel, Field


class VectorStoreConfigResponse(BaseModel):
    backend: str
    json_index_path: str
    project_id: int
    qdrant_url: str
    qdrant_collection: str
    qdrant_prefer_grpc: bool
    qdrant_timeout_seconds: float
    qdrant_api_key_configured: bool


class VectorStoreSyncRequest(BaseModel):
    project_id: int = Field(default=1, gt=0)
    index_path: str = "outputs/vector_index.json"
    backend: str | None = None
    batch_size: int = Field(default=128, gt=0)
    delete_stale: bool = True
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "codedoc_chunks_v1"
    qdrant_api_key: str | None = None


class VectorStoreCountRequest(BaseModel):
    project_id: int = Field(default=1, gt=0)
    backend: str | None = None
    index_path: str = "outputs/vector_index.json"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "codedoc_chunks_v1"
    qdrant_api_key: str | None = None
