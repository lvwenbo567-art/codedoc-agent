from __future__ import annotations

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langgraph_agent.tool_call_normalizer import normalize_tool_call


def test_normalize_read_file_alias_and_arguments() -> None:
    result = normalize_tool_call(
        {
            "id": "call_1",
            "name": "read_file",
            "args": {
                "path": "test_project/search.py",
                "start": 8,
                "end": 23,
            },
        }
    )

    assert result == {
        "id": "call_1",
        "name": "read_file_range",
        "args": {
            "source_path": "test_project/search.py",
            "start_line": 8,
            "end_line": 23,
        },
    }


def test_normalize_symbol_alias_and_arguments() -> None:
    result = normalize_tool_call(
        {
            "id": "call_2",
            "name": "find_symbol",
            "args": {
                "function_name": "keyword_score",
            },
        }
    )

    assert result == {
        "id": "call_2",
        "name": "get_symbol_definition",
        "args": {
            "symbol_name": "keyword_score",
        },
    }


def test_normalize_pytest_alias_and_arguments() -> None:
    result = normalize_tool_call(
        {
            "id": "call_3",
            "name": "run_tests",
            "args": {
                "path": "tests/test_search.py",
                "timeout": 30,
            },
        }
    )

    assert result == {
        "id": "call_3",
        "name": "run_project_tests",
        "args": {
            "test_path": "tests/test_search.py",
            "max_seconds": 30,
        },
    }
