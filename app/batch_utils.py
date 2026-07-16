from typing import List, TypeVar


T = TypeVar("T")


def split_batches(
    items: List[T],
    batch_size: int,
) -> List[List[T]]:
    """
    将列表按 batch_size 拆分成多个批次。
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size 必须大于 0"
        )

    return [
        items[start:start + batch_size]
        for start in range(
            0,
            len(items),
            batch_size,
        )
    ]