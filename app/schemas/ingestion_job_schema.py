from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictIngestionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateIngestionJobRequest(StrictIngestionModel):
    project_id: int = Field(ge=1)
    project_root: str = Field(min_length=1, max_length=1000)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    upsert_batch_size: int = Field(default=128, ge=1, le=2000)


class IngestionJobResponse(StrictIngestionModel):
    job_id: str
    project_id: int
    status: str
    stage: str
    progress: float
    attempt: int
    request_data: dict[str, Any]
    result_data: dict[str, Any]
    parent_job_id: str | None
    cancel_requested: bool
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    error_type: str | None
    error_message: str | None
