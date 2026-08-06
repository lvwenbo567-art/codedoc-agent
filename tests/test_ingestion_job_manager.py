from __future__ import annotations

import asyncio

import pytest

from ingestion.ingestion_job_manager import IngestionJobManager
from ingestion.ingestion_job_models import IngestionStage, IngestionJobStatus
from ingestion.ingestion_job_repository import IngestionJobRepository


class SuccessRunner:
    async def run(self, *, job_id, request_data, progress_callback):
        await progress_callback(IngestionStage.EMBEDDING, 0.6, {"job_id": job_id})
        return {"ok": True}


class SlowRunner:
    async def run(self, *, job_id, request_data, progress_callback):
        await progress_callback(IngestionStage.SCANNING, 0.1, None)
        await asyncio.sleep(10)
        return {}


async def wait_finished(repo, job_id):
    for _ in range(100):
        job = await repo.get_job(job_id)
        if job.status in {IngestionJobStatus.SUCCEEDED, IngestionJobStatus.FAILED, IngestionJobStatus.CANCELLED}:
            return job
        await asyncio.sleep(0.01)
    raise AssertionError("job not finished")


@pytest.mark.asyncio
async def test_manager_success_retry_and_cancel(tmp_path) -> None:
    repo = IngestionJobRepository(database_path=str(tmp_path / "jobs.sqlite")); await repo.start()
    manager = IngestionJobManager(repository=repo, runner=SuccessRunner())
    await manager.start()
    job = await manager.create_job(project_id=1, request_data={})
    assert (await wait_finished(repo, job.job_id)).status == IngestionJobStatus.SUCCEEDED
    await manager.close(); await repo.close()

    repo = IngestionJobRepository(database_path=str(tmp_path / "cancel.sqlite")); await repo.start()
    manager = IngestionJobManager(repository=repo, runner=SlowRunner())
    job = await manager.create_job(project_id=1, request_data={})
    await asyncio.sleep(0)
    await manager.cancel_job(job.job_id)
    cancelled = await wait_finished(repo, job.job_id)
    assert cancelled.status == IngestionJobStatus.CANCELLED
    retry = await manager.retry_job(job.job_id)
    assert retry.parent_job_id == job.job_id and retry.attempt == 2
    await manager.close(); await repo.close()
