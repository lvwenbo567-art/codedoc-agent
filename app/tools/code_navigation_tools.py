from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.code_navigation_models import GetSymbolDefinitionArgs, ReadFileRangeArgs
from tools.errors import ToolBusinessError
from tools.registry import ToolRegistry, ToolSpec


ALLOWED_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
}


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip()


def _resolve_safe_project_file(
    *,
    project_root: Path,
    source_path: str,
) -> Path:
    """
    将模型提供的相对路径解析为项目内文件。

    防止：
    - 绝对路径读取
    - ../ 路径穿越
    - 读取项目目录之外的文件
    """
    normalized = _normalize_path(source_path)
    relative_path = Path(normalized)

    if relative_path.is_absolute():
        raise ToolBusinessError(
            error_code="ABSOLUTE_PATH_FORBIDDEN",
            message="source_path 必须是相对于项目根目录的路径",
        )

    root = project_root.resolve()
    candidate = (root / relative_path).resolve()

    if candidate != root and root not in candidate.parents:
        raise ToolBusinessError(
            error_code="PATH_OUTSIDE_PROJECT",
            message="禁止读取项目根目录之外的文件",
        )

    if not candidate.exists():
        raise ToolBusinessError(
            error_code="SOURCE_FILE_NOT_FOUND",
            message=f"文件不存在：{normalized}",
        )

    if not candidate.is_file():
        raise ToolBusinessError(
            error_code="SOURCE_PATH_NOT_FILE",
            message=f"路径不是文件：{normalized}",
        )

    if candidate.suffix.lower() not in ALLOWED_TEXT_SUFFIXES:
        raise ToolBusinessError(
            error_code="UNSUPPORTED_SOURCE_SUFFIX",
            message=f"不允许读取该文件类型：{candidate.suffix}",
        )

    return candidate


def _load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    if not chunks_path.exists():
        raise ToolBusinessError(
            error_code="CHUNKS_NOT_FOUND",
            message=f"Chunk 文件不存在：{chunks_path}",
        )

    try:
        data = json.loads(
            chunks_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ToolBusinessError(
            error_code="INVALID_CHUNKS_JSON",
            message=f"Chunk 文件不是合法 JSON：{exc}",
        ) from exc

    if isinstance(data, dict):
        chunks = data.get("chunks") or data.get("records") or data.get("data")
    else:
        chunks = data

    if not isinstance(chunks, list):
        raise ToolBusinessError(
            error_code="INVALID_CHUNKS_FORMAT",
            message="Chunk 文件顶层必须是列表，或包含 chunks/records/data 列表字段",
        )

    return [
        chunk
        for chunk in chunks
        if isinstance(chunk, dict)
    ]


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False

    return text[:max_chars] + "\n...[content truncated]", True


def _content_hash(content: str) -> str:
    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()[:16]


def _read_file_range(
    *,
    project_root: str,
    source_path: str,
    start_line: int,
    end_line: int,
    max_chars: int,
) -> dict[str, Any]:
    file_path = _resolve_safe_project_file(
        project_root=Path(project_root),
        source_path=source_path,
    )

    lines = file_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    total_lines = len(lines)

    if start_line > total_lines:
        raise ToolBusinessError(
            error_code="START_LINE_OUT_OF_RANGE",
            message=(
                f"start_line 超出文件总行数："
                f"{start_line} > {total_lines}"
            ),
        )

    actual_end_line = min(end_line, total_lines)
    selected_lines = lines[start_line - 1:actual_end_line]
    numbered_content = "\n".join(
        f"{line_number}: {line}"
        for line_number, line in enumerate(
            selected_lines,
            start=start_line,
        )
    )
    content, truncated = _truncate_text(
        numbered_content,
        max_chars,
    )

    return {
        "source_path": _normalize_path(source_path),
        "absolute_path": str(file_path),
        "start_line": start_line,
        "end_line": actual_end_line,
        "requested_end_line": end_line,
        "total_lines": total_lines,
        "line_count": len(selected_lines),
        "truncated": truncated,
        "content_hash": _content_hash(content),
        "content": content,
    }


def _symbol_candidates(
    *,
    chunks: list[dict[str, Any]],
    symbol_name: str,
    source_path: str | None,
    exact_match: bool,
) -> list[dict[str, Any]]:
    normalized_symbol = symbol_name.strip()
    short_symbol = normalized_symbol.split(".")[-1]
    normalized_source_path = (
        _normalize_path(source_path)
        if source_path
        else None
    )
    candidates: list[dict[str, Any]] = []

    for chunk in chunks:
        if chunk.get("chunk_type") != "code":
            continue

        chunk_source_path = _normalize_path(
            str(chunk.get("source_path") or "")
        )

        if normalized_source_path and chunk_source_path != normalized_source_path:
            continue

        symbol_fields = [
            chunk.get("symbol_name"),
            chunk.get("qualified_name"),
            chunk.get("signature"),
            chunk.get("chunk_id"),
        ]
        content = str(chunk.get("content") or "")
        searchable_values = [
            str(value)
            for value in symbol_fields
            if value
        ]

        score = 0
        match_type = "none"

        if exact_match:
            if normalized_symbol in searchable_values:
                score = 100
                match_type = "exact"
            elif short_symbol in searchable_values:
                score = 90
                match_type = "short_exact"
            elif any(
                str(value).endswith(f".{short_symbol}")
                for value in searchable_values
            ):
                score = 80
                match_type = "qualified_suffix"

        if score == 0:
            lowered_symbol = normalized_symbol.lower()
            lowered_short = short_symbol.lower()
            lowered_values = [
                value.lower()
                for value in searchable_values
            ]
            lowered_content = content.lower()

            if any(lowered_symbol in value for value in lowered_values):
                score = 70
                match_type = "symbol_contains"
            elif any(lowered_short in value for value in lowered_values):
                score = 60
                match_type = "short_contains"
            elif lowered_symbol in lowered_content:
                score = 40
                match_type = "content_contains"
            elif lowered_short in lowered_content:
                score = 30
                match_type = "content_short_contains"

        if score <= 0:
            continue

        item = dict(chunk)
        item["_symbol_match_score"] = score
        item["_symbol_match_type"] = match_type
        candidates.append(item)

    candidates.sort(
        key=lambda item: (
            item.get("_symbol_match_score", 0),
            -(item.get("start_line") or 10**9),
            item.get("source_path") or "",
        ),
        reverse=True,
    )

    return candidates


def _compact_symbol_definition(
    *,
    chunk: dict[str, Any],
    rank: int,
    max_content_chars: int,
) -> dict[str, Any]:
    content = str(chunk.get("content") or "")
    truncated_content, truncated = _truncate_text(
        content,
        max_content_chars,
    )

    return {
        "rank": rank,
        "chunk_id": chunk.get("chunk_id"),
        "source_path": chunk.get("source_path"),
        "source_name": chunk.get("source_name"),
        "source_suffix": chunk.get("source_suffix"),
        "chunk_type": chunk.get("chunk_type"),
        "code_unit_type": chunk.get("code_unit_type"),
        "symbol_name": chunk.get("symbol_name"),
        "qualified_name": chunk.get("qualified_name"),
        "parent_class": chunk.get("parent_class"),
        "signature": chunk.get("signature"),
        "start_line": chunk.get("start_line"),
        "end_line": chunk.get("end_line"),
        "docstring": chunk.get("docstring"),
        "match_type": chunk.get("_symbol_match_type"),
        "match_score": chunk.get("_symbol_match_score"),
        "truncated": truncated,
        "content_hash": _content_hash(truncated_content),
        "content": truncated_content,
    }


def _get_symbol_definition(
    *,
    chunks_path: str,
    symbol_name: str,
    source_path: str | None,
    exact_match: bool,
    max_results: int,
    max_content_chars: int,
) -> dict[str, Any]:
    chunks = _load_chunks(Path(chunks_path))
    candidates = _symbol_candidates(
        chunks=chunks,
        symbol_name=symbol_name,
        source_path=source_path,
        exact_match=exact_match,
    )
    selected = candidates[:max_results]

    grouped_by_file: dict[str, int] = defaultdict(int)
    for item in selected:
        grouped_by_file[
            _normalize_path(str(item.get("source_path") or ""))
        ] += 1

    return {
        "symbol_name": symbol_name,
        "source_path": source_path,
        "exact_match": exact_match,
        "result_count": len(selected),
        "candidate_count": len(candidates),
        "grouped_by_file": dict(grouped_by_file),
        "results": [
            _compact_symbol_definition(
                chunk=chunk,
                rank=rank,
                max_content_chars=max_content_chars,
            )
            for rank, chunk in enumerate(selected, start=1)
        ],
    }


def register_code_navigation_tools(
    *,
    registry: ToolRegistry,
    project_root: str,
    chunks_path: str,
) -> None:
    """
    注册确定性代码导航工具。

    这些工具不依赖模型推理，主要负责精确读取源码和 AST Chunk。
    """

    def read_file_range(
        source_path: str,
        start_line: int = 1,
        end_line: int = 1,
        max_chars: int = 20000,
    ) -> dict[str, Any]:
        return _read_file_range(
            project_root=project_root,
            source_path=source_path,
            start_line=start_line,
            end_line=end_line,
            max_chars=max_chars,
        )

    def get_symbol_definition(
        symbol_name: str,
        source_path: str | None = None,
        exact_match: bool = True,
        max_results: int = 5,
        max_content_chars: int = 12000,
    ) -> dict[str, Any]:
        return _get_symbol_definition(
            chunks_path=chunks_path,
            symbol_name=symbol_name,
            source_path=source_path,
            exact_match=exact_match,
            max_results=max_results,
            max_content_chars=max_content_chars,
        )

    registry.register(
        ToolSpec(
            name="read_file_range",
            description=(
                "根据相对文件路径和行号读取项目内真实源码或文档片段。"
                "当检索结果提供 source_path、start_line、end_line 后，"
                "用它查看更完整上下文。"
            ),
            args_model=ReadFileRangeArgs,
            handler=read_file_range,
        )
    )
    registry.register(
        ToolSpec(
            name="get_symbol_definition",
            description=(
                "根据函数名、类名或方法名从 AST Chunk 中精确查找定义。"
                "适合 RerankClient.score、EmbeddingClient、keyword_score "
                "这类明确代码标识符问题。"
            ),
            args_model=GetSymbolDefinitionArgs,
            handler=get_symbol_definition,
        )
    )

