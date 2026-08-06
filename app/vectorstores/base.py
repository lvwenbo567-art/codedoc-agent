from __future__ import annotations

from abc import ABC, abstractmethod

from vectorstores.models import (
    VectorDeleteResult,
    VectorPoint,
    VectorSearchFilters,
    VectorSearchResult,
    VectorUpsertResult,
)


class VectorStoreError(RuntimeError):
    """VectorStore 通用异常。"""


class VectorStoreConfigurationError(VectorStoreError):
    """VectorStore 配置异常。"""


class VectorDimensionMismatchError(VectorStoreError):
    """向量维度不一致。"""


class VectorStore(ABC):
    """统一向量存储接口，上层不关心 JSON 还是 Qdrant。"""

    @abstractmethod
    def ensure_ready(self, *, vector_size: int) -> None:
        """确保后端已准备好，并检查向量维度。"""

    @abstractmethod
    def upsert(self, *, points: list[VectorPoint], batch_size: int = 128) -> VectorUpsertResult:
        """新增或覆盖向量 Point。"""

    @abstractmethod
    def search(
        self,
        *,
        project_id: int,
        query_vector: list[float],
        top_k: int,
        filters: VectorSearchFilters | None = None,
    ) -> list[VectorSearchResult]:
        """在指定项目内执行向量搜索。"""

    @abstractmethod
    def delete_chunks(self, *, project_id: int, chunk_ids: list[str]) -> VectorDeleteResult:
        """删除指定项目中的 chunks。"""

    @abstractmethod
    def delete_project(self, *, project_id: int) -> VectorDeleteResult:
        """删除指定项目的全部向量。"""

    @abstractmethod
    def count(self, *, project_id: int) -> int:
        """统计指定项目的 Point 数量。"""

    @abstractmethod
    def list_chunk_ids(self, *, project_id: int) -> set[str]:
        """列出指定项目已有的 chunk_id。"""

    @abstractmethod
    def close(self) -> None:
        """释放后端资源。"""
