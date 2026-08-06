from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from ingestion.ingestion_job_manager import IngestionJobManager
from ingestion.ingestion_job_repository import IngestionJobNotFoundError
from schemas.ingestion_job_schema import CreateIngestionJobRequest, IngestionJobResponse

router = APIRouter(prefix="/ingestion/jobs", tags=["ingestion-jobs"])


def get_job_manager(request: Request) -> IngestionJobManager:
    manager = getattr(request.app.state, "ingestion_job_manager", None)
    if manager is None:
        raise RuntimeError("IngestionJobManager 尚未初始化")
    return manager


@router.post("", response_model=IngestionJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_ingestion_job(body: CreateIngestionJobRequest, request: Request) -> IngestionJobResponse:
    job = await get_job_manager(request).create_job(project_id=body.project_id, request_data=body.model_dump())
    return IngestionJobResponse.model_validate(job.model_dump())


@router.get("/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion_job(job_id: str, request: Request) -> IngestionJobResponse:
    try:
        return IngestionJobResponse.model_validate((await get_job_manager(request).repository.get_job(job_id)).model_dump())
    except IngestionJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{job_id}/cancel", response_model=IngestionJobResponse)
async def cancel_ingestion_job(job_id: str, request: Request) -> IngestionJobResponse:
    try:
        return IngestionJobResponse.model_validate((await get_job_manager(request).cancel_job(job_id)).model_dump())
    except IngestionJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{job_id}/retry", response_model=IngestionJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def retry_ingestion_job(job_id: str, request: Request) -> IngestionJobResponse:
    try:
        return IngestionJobResponse.model_validate((await get_job_manager(request).retry_job(job_id)).model_dump())
    except IngestionJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
