from __future__ import annotations

import asyncio
import uuid
from typing import Any

from ingestion.ingestion_job_models import IngestionJobRecord, IngestionJobStatus, IngestionStage, utc_now_iso
from ingestion.ingestion_job_repository import IngestionJobRepository
from ingestion.ingestion_job_runner import IngestionJobRunner


class IngestionJobManager:
    """SQLite 状态 + 当前进程 asyncio.Task 的单机 Job 管理器。"""

    def __init__(self, *, repository: IngestionJobRepository, runner: IngestionJobRunner, max_running_jobs: int = 2) -> None:
        if max_running_jobs <= 0: raise ValueError("max_running_jobs 必须大于 0")
        self.repository, self.runner = repository, runner
        self._job_semaphore = asyncio.Semaphore(max_running_jobs)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        '''{
    "job-a": <Task running>,
    "job-b": <Task running>,
}'''
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        await self.repository.mark_orphaned_jobs_failed()

    async def create_job(self, *, project_id: int, request_data: dict[str, Any], attempt: int = 1,
                         parent_job_id: str | None = None) -> IngestionJobRecord:
        job_id = str(uuid.uuid4())
        record = await self.repository.create_job(job_id=job_id, project_id=project_id, request_data=request_data,
                                                  attempt=attempt, parent_job_id=parent_job_id)
        task = asyncio.create_task(self._execute_job(job_id=job_id, request_data=request_data), name=f"ingestion-job-{job_id}")
        async with self._lock:
            self._tasks[job_id] = task#把 Task 存进内存字典
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))#Task 完成后清理内存引用
        return record

    async def _execute_job(self, *, job_id: str, request_data: dict[str, Any]) -> None:
        async with self._job_semaphore:
            await self.repository.update_state(job_id=job_id, status=IngestionJobStatus.RUNNING, started_at=utc_now_iso())
            async def progress_callback(stage: IngestionStage, progress: float, metadata: dict[str, Any] | None) -> None:
                job = await self.repository.get_job(job_id)
                if job.cancel_requested: raise asyncio.CancelledError#检查是否被取消
                await self.repository.update_state(job_id=job_id, stage=stage, progress=progress,
                                                   result_data=metadata if metadata is not None else job.result_data)
            try:
                result = await self.runner.run(job_id=job_id, request_data=request_data, progress_callback=progress_callback)
                # 某些底层同步操作无法立即响应 task.cancel()；结束后再次确认取消标记。
                if (await self.repository.get_job(job_id)).cancel_requested:
                    raise asyncio.CancelledError
                await self.repository.update_state(job_id=job_id, status=IngestionJobStatus.SUCCEEDED,
                                                   stage=IngestionStage.COMPLETED, progress=1.0, result_data=result, finished_at=utc_now_iso())
            except asyncio.CancelledError:
                await self.repository.update_state(job_id=job_id, status=IngestionJobStatus.CANCELLED, finished_at=utc_now_iso(),
                                                   error_type="CancelledError", error_message="任务被用户取消")
                raise
            except Exception as exc:
                await self.repository.update_state(job_id=job_id, status=IngestionJobStatus.FAILED, finished_at=utc_now_iso(),
                                                   error_type=type(exc).__name__, error_message=str(exc))

    async def cancel_job(self, job_id: str) -> IngestionJobRecord:
        job = await self.repository.get_job(job_id)
        if job.status in {IngestionJobStatus.SUCCEEDED, IngestionJobStatus.FAILED, IngestionJobStatus.CANCELLED}:
            raise ValueError("已结束的 Job 不能取消")
        await self.repository.update_state(job_id=job_id, status=IngestionJobStatus.CANCELLING, cancel_requested=True)
        task = self._tasks.get(job_id)
        if task is not None: task.cancel()#再取消内存 Task
        # Task 可能在真正开始执行前就被取消，此时协程体没有机会捕获
        # CancelledError。主动落库终态，避免 Job 永远停在 cancelling。
        await self.repository.update_state(
            job_id=job_id,
            status=IngestionJobStatus.CANCELLED,
            finished_at=utc_now_iso(),
            error_type="CancelledError",
            error_message="任务被用户取消",
        )
        return await self.repository.get_job(job_id)

    async def retry_job(self, job_id: str) -> IngestionJobRecord:
        old = await self.repository.get_job(job_id)
        if old.status not in {IngestionJobStatus.FAILED, IngestionJobStatus.CANCELLED}:
            raise ValueError("只有 failed 或 cancelled 任务可以重试")
        return await self.create_job(project_id=old.project_id, request_data=old.request_data,
                                     attempt=old.attempt + 1, parent_job_id=old.job_id)

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks: task.cancel()
        if tasks: await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
