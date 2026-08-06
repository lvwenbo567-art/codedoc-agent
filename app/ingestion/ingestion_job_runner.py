from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from ingestion.ingestion_job_models import IngestionStage

ProgressCallback = Callable[[IngestionStage, float, dict[str, Any] | None], Awaitable[None]]


class IngestionJobRunner(Protocol):
    async def run(self, *, job_id: str, request_data: dict[str, Any], progress_callback: ProgressCallback) -> dict[str, Any]: ...
    '''
    只要一个对象具备符合要求的 async run(...) 方法，就可以被当作 IngestionJobRunner 使用。
    '''