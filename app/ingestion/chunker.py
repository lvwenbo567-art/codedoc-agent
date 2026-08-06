from __future__ import annotations

import json
import pprint
import tomllib
from pathlib import Path
from typing import Any, Dict, List

from ingestion.code_chunker import build_python_code_chunks
from utils.chunk_metadata import get_file_metadata
from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


CONFIG_AWARE_SUFFIXES = {
    ".json",
    ".toml",
    ".yaml",
    ".yml",
}


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


def is_config_aware_suffix(suffix: str) -> bool:
    """
    判断文件是否可以按配置块切分。
    """
    return suffix.lower() in CONFIG_AWARE_SUFFIXES


def _parse_config_content(
    *,
    content: str,
    suffix: str,
) -> Any:
    suffix = suffix.lower()

    if suffix == ".json":
        return json.loads(content)

    if suffix == ".toml":
        return tomllib.loads(content)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("当前环境未安装 PyYAML") from exc

        return yaml.safe_load(content)

    raise ValueError(f"不支持配置解析的文件类型：{suffix}")


def _format_config_value(value: Any) -> str:
    """
    将配置块格式化为稳定文本，方便 Embedding 和关键词检索。
    例如原始对象：

{
    "port": 5432,
    "host": "localhost",
}

格式化后可能是：

{'host': 'localhost', 'port': 5432}
    """
    return pprint.pformat(
        value,
        width=100,
        sort_dicts=True,
    )


def _iter_config_sections(data: Any) -> list[tuple[str, Any]]:
    if isinstance(data, dict):
        return [
            (str(key), value)
            for key, value in data.items()
        ]

    if isinstance(data, list):
        return [
            (f"item_{index}", value)
            for index, value in enumerate(data)
        ]

    return [("root", data)]


def build_config_file_chunks(
    project_file: Dict,
    chunk_size: int,
    overlap: int,
) -> List[Dict]:
    """
    按 JSON/YAML/TOML 的顶层配置块切分。

    解析失败时回退普通文本切分，避免单个配置文件阻断整个扫描流程。
    """
    validate_chunk_params(chunk_size, overlap)
    source_path, source_name, source_suffix, content = get_file_metadata(
        project_file
    )

    try:
        parsed = _parse_config_content(
            content=content,
            suffix=source_suffix,
        )
    except Exception as exc:
        chunks = build_text_file_chunks(
            project_file=project_file,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in chunks:
            chunk["parser"] = "config_fallback"
            chunk["parse_error"] = str(exc)

        return chunks

    chunks: list[dict] = []
    chunk_index = 0

    for section_name, section_value in _iter_config_sections(parsed):
        section_text = (
            f"{section_name} = "
            f"{_format_config_value(section_value)}"
        )
        parts = chunk_text(
            text=section_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for part_index, part in enumerate(parts):
            chunks.append(
                {
                    "chunk_id": (
                        f"{source_path}::config"
                        f"::{section_name}"
                        f"::part_{part_index}"
                    ),
                    "source_path": source_path,
                    "source_name": source_name,
                    "source_suffix": source_suffix,
                    "chunk_type": "document",
                    "code_unit_type": "config_section",
                    "symbol_name": section_name,
                    "qualified_name": section_name,
                    "parent_class": None,
                    "signature": None,
                    "start_line": None,
                    "end_line": None,
                    "docstring": "",
                    "parser": source_suffix.lower().lstrip("."),
                    "parse_error": None,
                    "chunk_index": chunk_index,
                    "part_index": part_index,
                    "part_count": len(parts),
                    "content": part,
                    "length": len(part),
                }
            )
            chunk_index += 1

    return chunks


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
        elif is_config_aware_suffix(suffix):
            file_chunks = build_config_file_chunks(
                project_file=project_file,
                chunk_size=chunk_size,
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
