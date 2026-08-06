from __future__ import annotations

from repositories.vector_store import load_vector_index_bundle
from vectorstores.base import VectorStore
from vectorstores.models import VectorPoint, VectorSyncResult


def sync_vector_index_to_store(
    *,
    project_id: int,
    index_path: str,
    vector_store: VectorStore,
    batch_size: int = 128,
    delete_stale: bool = True,
) -> VectorSyncResult:
    """把现有 vector_index.json 同步到统一 VectorStore。"""
    if project_id <= 0:
        raise ValueError("project_id 必须大于 0")

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    bundle = load_vector_index_bundle(index_path)
    records = bundle["records"]

    if not records:
        raise ValueError("向量索引 records 不能为空")

    points = [
        VectorPoint.from_index_record(project_id=project_id, record=record)
        for record in records
    ]
    vector_size = points[0].dimension
    vector_store.ensure_ready(vector_size=vector_size)

    existing_chunk_ids = vector_store.list_chunk_ids(project_id=project_id)
    source_chunk_ids = {point.chunk_id for point in points}
    stale_chunk_ids = sorted(existing_chunk_ids - source_chunk_ids)

    upsert_result = vector_store.upsert(points=points, batch_size=batch_size)
    deleted_stale_count = 0

    if delete_stale and stale_chunk_ids:
        delete_result = vector_store.delete_chunks(
            project_id=project_id,
            chunk_ids=stale_chunk_ids,
        )
        deleted_stale_count = delete_result.deleted_count or len(stale_chunk_ids)

    final_count = vector_store.count(project_id=project_id)

    return VectorSyncResult(
        project_id=project_id,
        source_count=len(points),
        existing_count=len(existing_chunk_ids),
        upserted_count=upsert_result.upserted_count,
        deleted_stale_count=deleted_stale_count,
        final_count=final_count,
        vector_size=vector_size,
        stale_chunk_ids=stale_chunk_ids,
    )
