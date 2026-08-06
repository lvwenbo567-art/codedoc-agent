from __future__ import annotations

import json
import math
import tempfile
import threading
from pathlib import Path
from typing import Any

from vectorstores.base import VectorDimensionMismatchError, VectorStore
from vectorstores.models import (
    VectorDeleteResult,
    VectorPoint,
    VectorSearchFilters,
    VectorSearchResult,
    VectorUpsertResult,
)

RECORD_KEYS = ("records", "items", "vectors")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算余弦相似度。"""
    if len(left) != len(right):
        raise VectorDimensionMismatchError(
            f"余弦相似度计算时向量维度不一致：{len(left)} != {len(right)}"
        )

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def _matches_filters(payload: dict[str, Any], filters: VectorSearchFilters | None) -> bool:
    if filters is None:
        return True

    if filters.chunk_type is not None and payload.get("chunk_type") != filters.chunk_type:
        return False

    if filters.source_path is not None and payload.get("source_path") != filters.source_path:
        return False

    if filters.source_suffix is not None and payload.get("source_suffix") != filters.source_suffix:
        return False

    if filters.content_hash is not None and payload.get("content_hash") != filters.content_hash:
        return False

    if filters.embedding_model is not None and payload.get("embedding_model") != filters.embedding_model:
        return False

    if filters.chunk_ids and str(payload.get("chunk_id")) not in filters.chunk_ids:
        return False

    return True


class JsonVectorStore(VectorStore):
    """基于本地 JSON 文件的 VectorStore，用于开发、测试和降级。"""

    def __init__(self, *, index_path: str, project_id: int) -> None:
        if project_id <= 0:
            raise ValueError("project_id 必须大于 0")

        self.index_path = Path(index_path)
        self.project_id = project_id
        self._lock = threading.RLock()

    def _validate_project(self, project_id: int) -> None:
        if project_id != self.project_id:
            raise ValueError(
                f"当前 JSON VectorStore 绑定 project_id={self.project_id}，不能访问 project_id={project_id}"
            )

    def _load_document(self) -> tuple[Any, str | None, list[dict[str, Any]]]:
        if not self.index_path.exists():
            return {"metadata": {}, "records": []}, "records", []

        data = json.loads(self.index_path.read_text(encoding="utf-8"))

        if isinstance(data, list):
            return data, None, data

        if not isinstance(data, dict):
            raise ValueError("JSON 向量索引格式不正确")

        for key in RECORD_KEYS:
            records = data.get(key)
            if isinstance(records, list):
                return data, key, records

        data["records"] = []
        return data, "records", data["records"]

    def _save_document(self, document: Any, records_key: str | None, records: list[dict[str, Any]]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(document, list):
            payload = records
        else:
            payload = dict(document)
            payload[records_key or "records"] = records

        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self.index_path.parent),
            delete=False,
            suffix=".tmp",
        ) as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
            temp_path = Path(fp.name)

        temp_path.replace(self.index_path)

    def _records_to_points(self, records: list[dict[str, Any]]) -> list[VectorPoint]:
        return [
            VectorPoint.from_index_record(project_id=self.project_id, record=record)
            for record in records
        ]

    def ensure_ready(self, *, vector_size: int) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size 必须大于 0")

        with self._lock:
            _, _, records = self._load_document()

        if not records:
            return

        first = VectorPoint.from_index_record(project_id=self.project_id, record=records[0])

        if first.dimension != vector_size:
            raise VectorDimensionMismatchError(
                f"JSON 索引维度 {first.dimension} 与目标维度 {vector_size} 不一致"
            )

    def upsert(self, *, points: list[VectorPoint], batch_size: int = 128) -> VectorUpsertResult:
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        if not points:
            return VectorUpsertResult(received_count=0, upserted_count=0, batch_count=0)

        for point in points:
            self._validate_project(point.project_id)

        with self._lock:
            document, records_key, records = self._load_document()
            by_chunk_id = {str(record.get("chunk_id")): dict(record) for record in records}

            for point in points:
                by_chunk_id[point.chunk_id] = point.to_json_record()

            final_records = list(by_chunk_id.values())
            self._save_document(document=document, records_key=records_key, records=final_records)

        batch_count = (len(points) + batch_size - 1) // batch_size
        return VectorUpsertResult(
            received_count=len(points),
            upserted_count=len(points),
            batch_count=batch_count,
        )

    def search(
        self,
        *,
        project_id: int,
        query_vector: list[float],
        top_k: int,
        filters: VectorSearchFilters | None = None,
    ) -> list[VectorSearchResult]:
        self._validate_project(project_id)

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        with self._lock:
            _, _, records = self._load_document()
            points = self._records_to_points(records)

        results: list[VectorSearchResult] = []

        for point in points:
            if not _matches_filters(point.payload, filters):
                continue

            score = cosine_similarity(query_vector, point.vector)
            results.append(
                VectorSearchResult(
                    point_id=point.point_id,
                    project_id=point.project_id,
                    chunk_id=point.chunk_id,
                    score=score,
                    payload=point.payload,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def delete_chunks(self, *, project_id: int, chunk_ids: list[str]) -> VectorDeleteResult:
        self._validate_project(project_id)
        target_ids = {str(chunk_id) for chunk_id in chunk_ids}

        with self._lock:
            document, records_key, records = self._load_document()
            kept_records = [record for record in records if str(record.get("chunk_id")) not in target_ids]
            deleted_count = len(records) - len(kept_records)
            self._save_document(document=document, records_key=records_key, records=kept_records)

        return VectorDeleteResult(deleted_count=deleted_count, requested_count=len(target_ids))

    def delete_project(self, *, project_id: int) -> VectorDeleteResult:
        self._validate_project(project_id)

        with self._lock:
            document, records_key, records = self._load_document()
            deleted_count = len(records)
            self._save_document(document=document, records_key=records_key, records=[])

        return VectorDeleteResult(deleted_count=deleted_count, requested_count=None)

    def count(self, *, project_id: int) -> int:
        self._validate_project(project_id)

        with self._lock:
            _, _, records = self._load_document()

        return len(records)

    def list_chunk_ids(self, *, project_id: int) -> set[str]:
        self._validate_project(project_id)

        with self._lock:
            _, _, records = self._load_document()

        return {str(record.get("chunk_id")) for record in records if record.get("chunk_id")}

    def close(self) -> None:
        return None
