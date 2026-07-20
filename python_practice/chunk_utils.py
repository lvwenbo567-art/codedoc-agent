from collections.abc import Iterator, Sequence
import hashlib
from typing import Any, TypeVar


T = TypeVar("T")


def filter_chunks(
    chunks: list[dict[str, Any]],
    chunk_type: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    按 chunk_type 和 limit 过滤 chunk 列表。
    """
    if limit is not None and limit < 0:
        raise ValueError("limit 不能小于 0")

    results = chunks

    if chunk_type is not None:
        results = [
            chunk
            for chunk in results
            if chunk.get("chunk_type") == chunk_type
        ]

    if limit is not None:
        results = results[:limit]

    return results


def iter_batches(
    items: Sequence[T],
    batch_size: int,
) -> Iterator[list[T]]:
    """
    使用生成器按 batch_size 分批返回数据。
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def compute_content_hash(content: str) -> str:
    """
    计算文本内容的 SHA256 哈希值，用于判断内容是否变化。
    """
    normalized_content = content.replace("\r\n", "\n").strip()

    return hashlib.sha256(
        normalized_content.encode("utf-8")
    ).hexdigest()


def calculate_chunk_stats(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    统计 chunk 总数、类型分布和平均长度。
    """
    type_counts: dict[str, int] = {}

    for chunk in chunks:
        chunk_type = chunk.get("chunk_type", "unknown")
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

    if not chunks:
        average_length = 0
    else:
        average_length = sum(
            len(chunk.get("content", ""))
            for chunk in chunks
        ) / len(chunks)

    return {
        "total": len(chunks),
        "type_counts": type_counts,
        "average_length": average_length,
    }
