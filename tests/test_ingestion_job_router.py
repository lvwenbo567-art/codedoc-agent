from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.ingestion_job_router import router
from ingestion.ingestion_job_models import IngestionJobRecord, IngestionJobStatus, IngestionStage


def record(job_id: str = "job", status: IngestionJobStatus = IngestionJobStatus.PENDING) -> IngestionJobRecord:
    return IngestionJobRecord(job_id=job_id, project_id=1, status=status, stage=IngestionStage.QUEUED,
                              progress=0, attempt=1, request_data={"project_root": "."}, created_at="now", updated_at="now")


class Manager:
    def __init__(self): self.repository = self
    async def create_job(self, **_): return record()
    async def get_job(self, job_id): return record(job_id)
    async def cancel_job(self, job_id): return record(job_id, IngestionJobStatus.CANCELLED)
    async def retry_job(self, job_id): return record("retry")


def test_router_create_get_cancel_retry() -> None:
    app = FastAPI(); app.include_router(router); app.state.ingestion_job_manager = Manager()
    client = TestClient(app)
    created = client.post("/ingestion/jobs", json={"project_id": 1, "project_root": "."})
    assert created.status_code == 202 and created.json()["job_id"] == "job"
    assert client.get("/ingestion/jobs/job").status_code == 200
    assert client.post("/ingestion/jobs/job/cancel").json()["status"] == "cancelled"
    assert client.post("/ingestion/jobs/job/retry").status_code == 202
