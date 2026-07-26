from pathlib import Path
import json
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from tools.code_navigation_tools import register_code_navigation_tools
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry


def build_navigation_registry(
    *,
    project_root: Path,
    chunks_path: Path,
) -> ToolRegistry:
    registry = ToolRegistry()

    register_code_navigation_tools(
        registry=registry,
        project_root=str(project_root),
        chunks_path=str(chunks_path),
    )

    return registry


def test_read_file_range(tmp_path):
    source_file = tmp_path / "example.py"
    source_file.write_text(
        "\n".join(
            [
                "line one",
                "line two",
                "line three",
                "line four",
            ]
        ),
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text("[]", encoding="utf-8")
    registry = build_navigation_registry(
        project_root=tmp_path,
        chunks_path=chunks_path,
    )

    result = ToolExecutor(registry).execute(
        tool_name="read_file_range",
        arguments={
            "source_path": "example.py",
            "start_line": 2,
            "end_line": 3,
            "max_chars": 1000,
        },
    )

    assert result.success is True
    assert result.data["start_line"] == 2
    assert result.data["end_line"] == 3
    assert "2: line two" in result.data["content"]
    assert "3: line three" in result.data["content"]


def test_read_file_range_normalizes_null_and_too_small_max_chars(tmp_path):
    source_file = tmp_path / "example.py"
    source_file.write_text(
        "\n".join(
            [
                "def keyword_score(query, text):",
                "    terms = query.split()",
                "    return sum(text.count(term) for term in terms)",
            ]
        ),
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text("[]", encoding="utf-8")
    registry = build_navigation_registry(
        project_root=tmp_path,
        chunks_path=chunks_path,
    )
    executor = ToolExecutor(registry)

    null_result = executor.execute(
        tool_name="read_file_range",
        arguments={
            "source_path": "example.py",
            "start_line": 1,
            "end_line": 3,
            "max_chars": None,
        },
    )

    small_result = executor.execute(
        tool_name="read_file_range",
        arguments={
            "source_path": "example.py",
            "start_line": 1,
            "end_line": 3,
            "max_chars": 200,
        },
    )

    assert null_result.success is True
    assert null_result.data["truncated"] is False
    assert "return sum" in null_result.data["content"]

    assert small_result.success is True
    assert small_result.data["truncated"] is False
    assert "return sum" in small_result.data["content"]


def test_read_file_rejects_path_traversal(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text("[]", encoding="utf-8")
    registry = build_navigation_registry(
        project_root=tmp_path,
        chunks_path=chunks_path,
    )

    result = ToolExecutor(registry).execute(
        tool_name="read_file_range",
        arguments={
            "source_path": "../outside.py",
            "start_line": 1,
            "end_line": 10,
            "max_chars": 1000,
        },
    )

    assert result.success is False
    assert result.error_code == "PATH_OUTSIDE_PROJECT"


def test_read_file_rejects_unsupported_suffix(tmp_path):
    source_file = tmp_path / "image.png"
    source_file.write_bytes(b"fake")
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text("[]", encoding="utf-8")
    registry = build_navigation_registry(
        project_root=tmp_path,
        chunks_path=chunks_path,
    )

    result = ToolExecutor(registry).execute(
        tool_name="read_file_range",
        arguments={
            "source_path": "image.png",
            "start_line": 1,
            "end_line": 1,
            "max_chars": 1000,
        },
    )

    assert result.success is False
    assert result.error_code == "UNSUPPORTED_SOURCE_SUFFIX"


def test_get_symbol_definition_exact(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    chunks = [
        {
            "chunk_id": "app/example.py::Calculator.add::part_0",
            "source_path": "app/example.py",
            "source_name": "example.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "code_unit_type": "method",
            "symbol_name": "add",
            "qualified_name": "Calculator.add",
            "parent_class": "Calculator",
            "signature": "def add(self, a, b):",
            "start_line": 10,
            "end_line": 12,
            "docstring": "",
            "content": "def add(self, a, b):\n    return a + b",
        },
        {
            "chunk_id": "docs/readme.md::document::0",
            "source_path": "docs/readme.md",
            "chunk_type": "document",
            "content": "Calculator.add usage",
        },
    ]
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8",
    )
    registry = build_navigation_registry(
        project_root=tmp_path,
        chunks_path=chunks_path,
    )

    result = ToolExecutor(registry).execute(
        tool_name="get_symbol_definition",
        arguments={
            "symbol_name": "Calculator.add",
            "exact_match": True,
            "max_results": 5,
            "max_content_chars": 1000,
        },
    )

    assert result.success is True
    assert result.data["result_count"] == 1
    assert result.data["results"][0]["qualified_name"] == "Calculator.add"
    assert result.data["results"][0]["match_type"] == "exact"


def test_code_doc_registry_contains_navigation_tools(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text("[]", encoding="utf-8")
    registry = build_navigation_registry(
        project_root=tmp_path,
        chunks_path=chunks_path,
    )

    assert "read_file_range" in registry.names()
    assert "get_symbol_definition" in registry.names()
