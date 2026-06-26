from typing import Dict, List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[str]:
    """
    按固定字符长度切分文本，并保留一定重叠。
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if overlap < 0:
        raise ValueError("overlap 不能小于 0")

    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def build_chunks(
    files: List[Dict],
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Dict]:
    """
    将项目文件切分为统一 chunk 结构。
    """
    results = []

    for file in files:
        content = file["content"]
        text_chunks = chunk_text(
            text=content,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for idx, chunk in enumerate(text_chunks):
            if file["suffix"] == ".py":
                chunk_type = "code"
            else:
                chunk_type = "document"

            results.append(
                {
                    "chunk_id": f"{file['path']}::chunk_{idx}",
                    "source_path": file["path"],
                    "source_name": file["name"],
                    "source_suffix": file["suffix"],
                    "chunk_type": chunk_type,
                    "chunk_index": idx,
                    "content": chunk,
                    "length": len(chunk),
                }
            )

    return results