from __future__ import annotations

import pytest

from ingestion.ingestion_job_models import IngestionJobStatus, IngestionStage
from ingestion.ingestion_job_repository import IngestionJobRepository


@pytest.mark.asyncio
async def test_repository_create_update_and_orphan_recovery(tmp_path) -> None:
    repository = IngestionJobRepository(database_path=str(tmp_path / "jobs.sqlite"))
    await repository.start()
    job = await repository.create_job(job_id="a", project_id=1, request_data={"project_root": "."})
    assert job.status == IngestionJobStatus.PENDING
    await repository.update_state(job_id="a", status=IngestionJobStatus.RUNNING, stage=IngestionStage.EMBEDDING, progress=0.5)
    assert (await repository.get_job("a")).stage == IngestionStage.EMBEDDING
    assert await repository.mark_orphaned_jobs_failed() == 1
    restored = await repository.get_job("a")
    assert restored.status == IngestionJobStatus.FAILED
    assert restored.error_type == "ProcessRestarted"
    await repository.close()
