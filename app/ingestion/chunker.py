from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ingestion.code_chunker import build_python_code_chunks
from utils.chunk_metadata import get_file_metadata
from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def validate_chunk_params(chunk_size: int, overlap: int) -> None:
    """
    校验 chunk 参数是否合法。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if overlap < 0:
        raise ValueError("overlap 不能小于 0")

    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    对普通文档进行固定长度字符切分。
    """
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")

    validate_chunk_params(chunk_size, overlap)

    if not text or not text.strip():
        return []

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


def get_chunk_type(suffix: str) -> str:
    """
    根据文件后缀判断 chunk 类型。
    """
    if suffix.lower() == ".py":
        return "code"

    return "document"


def build_text_file_chunks(
    project_file: Dict,
    chunk_size: int,
    overlap: int,
) -> List[Dict]:
    """
    构建普通文档 chunk，并补齐与 code chunk 一致的元数据字段。
    """
    validate_chunk_params(chunk_size, overlap)
    source_path, source_name, source_suffix, content = get_file_metadata(
        project_file
    )
    parts = chunk_text(
        text=content,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    chunks: list[dict] = []

    for chunk_index, part in enumerate(parts):
        chunks.append(
            {
                "chunk_id": f"{source_path}::document::{chunk_index}",
                "source_path": source_path,
                "source_name": source_name,
                "source_suffix": source_suffix,
                "chunk_type": "document",
                "code_unit_type": None,
                "symbol_name": None,
                "qualified_name": None,
                "parent_class": None,
                "signature": None,
                "start_line": None,
                "end_line": None,
                "docstring": "",
                "parser": "text",
                "parse_error": None,
                "chunk_index": chunk_index,
                "part_index": 0,
                "part_count": 1,
                "content": part,
                "length": len(part),
            }
        )

    return chunks


def build_chunks(
    files: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    """
    统一 Chunk 构建入口。

    Python 文件走 AST-aware code chunking；
    普通文档走固定长度字符切分。
    """
    if not isinstance(files, list):
        raise TypeError("files 必须是列表")

    validate_chunk_params(chunk_size, overlap)

    all_chunks = []

    for project_file in files:
        source_path = (
            project_file.get("path")
            or project_file.get("source_path")
            or ""
        )
        suffix = (
            project_file.get("suffix")
            or project_file.get("source_suffix")
            or Path(source_path).suffix
        )
        chunk_type = get_chunk_type(suffix)

        if chunk_type == "code":
            file_chunks = build_python_code_chunks(
                project_file=project_file,
                max_chunk_chars=chunk_size,
                overlap=overlap,
            )
        else:
            file_chunks = build_text_file_chunks(
                project_file=project_file,
                chunk_size=chunk_size,
                overlap=overlap,
            )

        all_chunks.extend(file_chunks)

    return all_chunks
