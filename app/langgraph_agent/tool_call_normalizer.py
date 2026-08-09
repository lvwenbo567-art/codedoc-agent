from __future__ import annotations

from typing import Any


TOOL_NAME_ALIASES = {
    "code_search": "search_code",
    "document_search": "search_documents",
    "search_docs": "search_documents",
    "search_document": "search_documents",
    "project_structure": "get_project_structure",
    "list_project_structure": "get_project_structure",
    "list_files": "get_project_structure",
    "read_file": "read_file_range",
    "read_source_file": "read_file_range",
    "open_file": "read_file_range",
    "cat_file": "read_file_range",
    "find_symbol": "get_symbol_definition",
    "locate_symbol": "get_symbol_definition",
    "search_symbol": "get_symbol_definition",
    "get_definition": "get_symbol_definition",
    "get_function_definition": "get_symbol_definition",
    "run_tests": "run_project_tests",
    "run_pytest": "run_project_tests",
    "pytest": "run_project_tests",
}


def normalize_tool_name(name: str) -> str:
    normalized = name.strip()
    return TOOL_NAME_ALIASES.get(normalized, normalized)


def _rename_first_existing(
    args: dict[str, Any],
    *,
    target_key: str,
    aliases: tuple[str, ...],
) -> None:
    if target_key in args:
        return

    for alias in aliases:
        if alias in args:
            args[target_key] = args.pop(alias)
            return


def normalize_tool_arguments(
    tool_name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(args)

    if tool_name == "read_file_range":
        _rename_first_existing(
            normalized,
            target_key="source_path",
            aliases=("path", "file_path", "file", "source"),
        )
        _rename_first_existing(
            normalized,
            target_key="start_line",
            aliases=("start", "line_start", "start_lineno"),
        )
        _rename_first_existing(
            normalized,
            target_key="end_line",
            aliases=("end", "line_end", "end_lineno"),
        )
        normalized.setdefault("start_line", 1)
        normalized.setdefault("end_line", 120)

    if tool_name == "get_symbol_definition":
        _rename_first_existing(
            normalized,
            target_key="symbol_name",
            aliases=("symbol", "name", "function_name", "class_name", "method_name"),
        )

    if tool_name == "run_project_tests":
        _rename_first_existing(
            normalized,
            target_key="test_path",
            aliases=("path", "file_path", "test_file", "test_dir"),
        )
        _rename_first_existing(
            normalized,
            target_key="max_seconds",
            aliases=("timeout", "timeout_seconds"),
        )

    return normalized


def normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    tool_name = normalize_tool_name(str(call.get("name") or ""))
    args = call.get("args") if isinstance(call.get("args"), dict) else {}

    return {
        "id": str(call.get("id") or ""),
        "name": tool_name,
        "args": normalize_tool_arguments(
            tool_name=tool_name,
            args=args,
        ),
    }


def normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_tool_call(call) for call in tool_calls]
