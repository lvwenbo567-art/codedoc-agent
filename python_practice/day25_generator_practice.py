from collections.abc import Iterator
from pathlib import Path


def iter_file_lines(
    file_path: str,
) -> Iterator[str]:
    """
    使用生成器逐行读取文件，适合处理较大的文本文件。
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    if not path.is_file():
        raise ValueError(f"路径不是文件：{file_path}")

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            yield line.rstrip("\n")


def iter_chunk_batches(
    chunks: list[dict],
    batch_size: int,
) -> Iterator[list[dict]]:
    """
    按批次逐步返回 chunks，避免一次性创建所有批次。
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):
        yield chunks[start:start + batch_size]


def filter_chunk_contents(
    chunks: list[dict],
    chunk_type: str,
) -> list[str]:
    """
    使用列表推导式筛选指定类型的 chunk 内容。
    """
    return [
        chunk["content"]
        for chunk in chunks
        if chunk.get("chunk_type") == chunk_type
    ]


def calculate_total_length(
    chunks: list[dict],
) -> int:
    """
    使用生成器表达式计算所有 chunks 的总长度。
    """
    return sum(
        chunk.get("length", 0)
        for chunk in chunks
    )
