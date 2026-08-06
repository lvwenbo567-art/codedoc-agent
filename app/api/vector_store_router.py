from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.api_response import success_response
from schemas.vector_store_schema import (
    VectorStoreCountRequest,
    VectorStoreSyncRequest,
)
from services.vector_store_sync_service import sync_vector_index_to_store
from vectorstores.config import VectorStoreConfig
from vectorstores.factory import create_vector_store

router = APIRouter(prefix="/vector-store", tags=["vector-store"])


def _build_store_config(
    *,
    backend: str | None,
    project_id: int,
    index_path: str,
    qdrant_url: str,
    qdrant_collection: str,
    qdrant_api_key: str | None,
) -> VectorStoreConfig:
    config = VectorStoreConfig.from_env()
    updates = {
        "project_id": project_id,
        "json_index_path": index_path,
        "qdrant_url": qdrant_url,
        "qdrant_collection": qdrant_collection,
        "qdrant_api_key": qdrant_api_key,
    }

    if backend is not None:
        updates["backend"] = backend

    return config.model_copy(update=updates)


@router.get("/config")
def get_vector_store_config() -> dict:
    """查看当前 VectorStore 环境配置。"""
    config = VectorStoreConfig.from_env()

    return success_response(
        data={
            "backend": config.backend,
            "json_index_path": config.json_index_path,
            "project_id": config.project_id,
            "qdrant_url": config.qdrant_url,
            "qdrant_collection": config.qdrant_collection,
            "qdrant_prefer_grpc": config.qdrant_prefer_grpc,
            "qdrant_timeout_seconds": config.qdrant_timeout_seconds,
            "qdrant_api_key_configured": bool(config.qdrant_api_key),
        }
    )


@router.post("/sync")
def sync_vector_store_api(request: VectorStoreSyncRequest) -> dict:
    """把现有 JSON vector_index 同步到配置的 VectorStore 后端。"""
    config = _build_store_config(
        backend=request.backend,
        project_id=request.project_id,
        index_path=request.index_path,
        qdrant_url=request.qdrant_url,
        qdrant_collection=request.qdrant_collection,
        qdrant_api_key=request.qdrant_api_key,
    )
    store = create_vector_store(config)

    try:
        result = sync_vector_index_to_store(
            project_id=request.project_id,
            index_path=request.index_path,
            vector_store=store,
            batch_size=request.batch_size,
            delete_stale=request.delete_stale,
        )
        return success_response(data=result.__dict__)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    finally:
        store.close()


@router.post("/count")
def count_vector_store_api(request: VectorStoreCountRequest) -> dict:
    """统计指定项目在当前 VectorStore 后端中的向量数量。"""
    config = _build_store_config(
        backend=request.backend,
        project_id=request.project_id,
        index_path=request.index_path,
        qdrant_url=request.qdrant_url,
        qdrant_collection=request.qdrant_collection,
        qdrant_api_key=request.qdrant_api_key,
    )
    store = create_vector_store(config)

    try:
        return success_response(
            data={
                "backend": config.backend,
                "project_id": request.project_id,
                "count": store.count(project_id=request.project_id),
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    finally:
        store.close()
