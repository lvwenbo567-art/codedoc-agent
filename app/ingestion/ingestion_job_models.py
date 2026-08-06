from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum#StrEnum 可以理解为“字符串形式的枚举”。
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestionJobStatus(StrEnum):#任务总体状态
    PENDING = "pending"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class IngestionStage(StrEnum):
    QUEUED = "queued"
    SCANNING = "scanning"
    LOADING = "loading"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    UPSERTING = "upserting"
    COMPLETED = "completed"


class IngestionJobRecord(BaseModel):#完整任务记录
    model_config = ConfigDict(extra="forbid")
    job_id: str#单次摄入任务唯一 ID；
    project_id: int#任务属于哪个代码仓库项目。
    status: IngestionJobStatus
    stage: IngestionStage
    progress: float = Field(ge=0, le=1)#进度
    attempt: int = Field(ge=1)
    request_data: dict[str, Any]
    result_data: dict[str, Any] = Field(default_factory=dict)
    parent_job_id: str | None = None
    cancel_requested: bool = False#用户是否请求取消。
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error_type: str | None = None
    error_message: str | None = None
