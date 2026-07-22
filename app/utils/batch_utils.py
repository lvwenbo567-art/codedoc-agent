from collections.abc import Iterator, Sequence
from typing import List, TypeVar


T = TypeVar("T")


def iter_batches(
    items: Sequence[T],
    batch_size: int,
) -> Iterator[list[T]]:
    """
    使用生成器逐批返回数据，避免一次性创建所有批次列表。
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    for start in range(
        0,
        len(items),
        batch_size,
    ):
        yield list(
            items[start:start + batch_size]
        )


def split_batches(
    items: Sequence[T],
    batch_size: int,
) -> List[List[T]]:
    """
    将列表按 batch_size 拆分成多个批次，保留旧接口兼容 Day24 代码。
    """
    return list(
        iter_batches(
            items=items,
            batch_size=batch_size,
        )
    )
