from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from vectorstores.point_id import build_vector_point_id


def _extract_vector(record: dict[str, Any]) -> list[float]:
    """从旧索引 record 中读取 embedding/vector，并校验为有限浮点数。"""
    raw_vector = record.get("embedding")

    if raw_vector is None:
        raw_vector = record.get("vector")

    if not isinstance(raw_vector, list):
        raise ValueError("向量记录缺少 embedding/vector")

    vector: list[float] = []

    for value in raw_vector:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("向量中包含非数字元素") from exc

        if not math.isfinite(number):
            raise ValueError("向量中不能包含 NaN 或 Inf")

        vector.append(number)

    if not vector:
        raise ValueError("向量不能为空")

    return vector


@dataclass(frozen=True)
class VectorPoint:
    """统一向量存储中的一条 Point。"""

    point_id: str
    project_id: int
    chunk_id: str
    vector: list[float]
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if self.project_id <= 0:
            raise ValueError("project_id 必须大于 0")

        if not self.chunk_id.strip():
            raise ValueError("chunk_id 不能为空")

        if not self.vector:
            raise ValueError("vector 不能为空")

        if not all(math.isfinite(float(value)) for value in self.vector):
            raise ValueError("vector 不能包含 NaN 或 Inf")

    @property
    def dimension(self) -> int:
        return len(self.vector)

    @classmethod
    def from_index_record(cls, *, project_id: int, record: dict[str, Any]) -> "VectorPoint":
        """把现有 JSON vector_index record 转成统一 VectorPoint。"""
        chunk_id_value = record.get("chunk_id")

        if chunk_id_value is None:
            raise ValueError("向量记录缺少 chunk_id")

        chunk_id = str(chunk_id_value).strip()
        vector = _extract_vector(record)
        payload = dict(record)
        payload.pop("embedding", None)
        payload.pop("vector", None)
        payload.pop("point_id", None)
        payload["project_id"] = project_id
        payload["chunk_id"] = chunk_id

        return cls(
            point_id=build_vector_point_id(project_id=project_id, chunk_id=chunk_id),
            project_id=project_id,
            chunk_id=chunk_id,
            vector=vector,
            payload=payload,
        )

    def to_json_record(self) -> dict[str, Any]:
        """转换回兼容旧 vector_index.json 的 record。"""
        record = dict(self.payload)
        record.update(
            {
                "point_id": self.point_id,
                "project_id": self.project_id,
                "chunk_id": self.chunk_id,
                "embedding": list(self.vector),
            }
        )
        return record


@dataclass(frozen=True)
class VectorSearchFilters:
    chunk_type: str | None = None
    source_path: str | None = None
    source_suffix: str | None = None
    content_hash: str | None = None
    embedding_model: str | None = None
    chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class VectorSearchResult:
    point_id: str
    project_id: int
    chunk_id: str
    score: float
    payload: dict[str, Any]

    def to_legacy_record(self, *, rank: int) -> dict[str, Any]:
        """转换为当前 vector_search_service / hybrid_search 使用的旧结果结构。"""
        result = dict(self.payload)
        result.update(
            {
                "rank": rank,
                "point_id": self.point_id,
                "project_id": self.project_id,
                "chunk_id": self.chunk_id,
                "score": self.score,
                "vector_score": self.score,
            }
        )
        return result


@dataclass(frozen=True)
class VectorUpsertResult:#写入了多少 Point、分了多少批
    received_count: int
    upserted_count: int
    batch_count: int


@dataclass(frozen=True)#请求删除多少、实际删除多少
class VectorDeleteResult:
    deleted_count: int | None
    requested_count: int | None


@dataclass(frozen=True)#JSON → Qdrant 同步任务的完整统计
class VectorSyncResult:
    project_id: int#同步的是哪个项目
    source_count: int#源 JSON 索引当前有多少 Chunk
    existing_count: int#Qdrant 同步前已有多少 Point
    upserted_count: int#本次写入/覆盖多少 Point
    deleted_stale_count: int#本次删除多少陈旧 Point
    final_count: int#同步后 Qdrant 最终还有多少 Point
    vector_size: int
    stale_chunk_ids: list[str] = field(default_factory=list)#被识别为陈旧、应删除的 Chunk ID 列表
