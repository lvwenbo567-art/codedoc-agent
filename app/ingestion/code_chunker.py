from __future__ import annotations

from utils.chunk_metadata import get_file_metadata
from ingestion.code_parser import PythonCodeParseError, parse_python_symbols


def split_code_content(
    content: str,
    max_chunk_chars: int,
    overlap: int,
) -> list[str]:
    """
    对超长函数、方法或模块内容做二次字符切分。
    """
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars 必须大于 0")

    if overlap < 0:
        raise ValueError("overlap 不能小于 0")

    if overlap >= max_chunk_chars:
        raise ValueError("overlap 必须小于 max_chunk_chars")

    if not content:
        return []

    parts: list[str] = []
    start = 0

    while start < len(content):
        end = min(start + max_chunk_chars, len(content))
        part = content[start:end]

        if part:
            parts.append(part)

        if end >= len(content):
            break

        start = end - overlap

    return parts


def build_fallback_code_chunks(
    project_file: dict,
    max_chunk_chars: int,
    overlap: int,
    parse_error: str,
) -> list[dict]:
    """
    AST 解析失败时退回普通字符切分，避免整个扫描流程失败。
    """
    source_path, source_name, source_suffix, content = get_file_metadata(project_file)
    content_parts = split_code_content(
        content=content,
        max_chunk_chars=max_chunk_chars,
        overlap=overlap,
    )
    chunks: list[dict] = []

    for chunk_index, part in enumerate(content_parts):
        chunks.append(
            {
                "chunk_id": f"{source_path}::fallback::{chunk_index}",
                "source_path": source_path,
                "source_name": source_name,
                "source_suffix": source_suffix,
                "chunk_type": "code",
                "code_unit_type": "text_fallback",
                "symbol_name": None,
                "qualified_name": None,
                "parent_class": None,
                "signature": None,
                "start_line": None,
                "end_line": None,
                "docstring": "",
                "parser": "text_fallback",
                "parse_error": parse_error,
                "chunk_index": chunk_index,
                "part_index": 0,
                "part_count": len(content_parts),
                "content": part,
                "length": len(part),
            }
        )

    return chunks


def build_python_code_chunks(
    project_file: dict,
    max_chunk_chars: int = 3000,
    overlap: int = 200,
) -> list[dict]:
    """
    把一个 Python 文件转换成 AST-aware code chunks。
    """
    source_path, source_name, source_suffix, content = get_file_metadata(project_file)

    try:
        symbols = parse_python_symbols(source=content, source_path=source_path)
    except PythonCodeParseError as exc:
        return build_fallback_code_chunks(
            project_file=project_file,
            max_chunk_chars=max_chunk_chars,
            overlap=overlap,
            parse_error=str(exc),
        )

    chunks: list[dict] = []

    for symbol_index, symbol in enumerate(symbols):
        content_parts = split_code_content(
            content=symbol["content"],
            max_chunk_chars=max_chunk_chars,
            overlap=overlap,
        )

        for part_index, part in enumerate(content_parts):
            chunk_index = len(chunks)
            qualified_name = symbol["qualified_name"]
            chunks.append(
                {
                    "chunk_id": (
                        f"{source_path}::{qualified_name}"
                        f"::part_{part_index}"
                    ),
                    "source_path": source_path,
                    "source_name": source_name,
                    "source_suffix": source_suffix,
                    "chunk_type": "code",
                    "code_unit_type": symbol["symbol_type"],
                    "symbol_name": symbol["symbol_name"],
                    "qualified_name": qualified_name,
                    "parent_class": symbol["parent_class"],
                    "signature": symbol["signature"],
                    "start_line": symbol["start_line"],
                    "end_line": symbol["end_line"],
                    "docstring": symbol["docstring"],
                    "parser": "python_ast",
                    "parse_error": None,
                    "symbol_index": symbol_index,
                    "chunk_index": chunk_index,
                    "part_index": part_index,
                    "part_count": len(content_parts),
                    "content": part,
                    "length": len(part),
                }
            )

    return chunks
