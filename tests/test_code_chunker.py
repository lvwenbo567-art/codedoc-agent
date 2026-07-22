from pathlib import Path
import sys

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))


from ingestion.code_chunker import build_python_code_chunks, split_code_content


def test_build_python_code_chunks_extracts_function_class_and_method():
    project_file = {
        "path": "app/example.py",
        "name": "example.py",
        "suffix": ".py",
        "content": '''
import os


CONSTANT = 1


class Calculator:
    """计算器。"""

    def add(self, a, b):
        """返回两个数之和。"""
        return a + b


def multiply(a, b):
    return a * b
''',
        "length": 180,
    }

    chunks = build_python_code_chunks(
        project_file=project_file,
        max_chunk_chars=1000,
        overlap=100,
    )

    qualified_names = {chunk["qualified_name"] for chunk in chunks}
    code_unit_types = {chunk["code_unit_type"] for chunk in chunks}

    assert "__module__" in qualified_names
    assert "Calculator" in qualified_names
    assert "Calculator.add" in qualified_names
    assert "multiply" in qualified_names
    assert {"module", "class", "method", "function"} <= code_unit_types

    method_chunk = next(
        chunk for chunk in chunks if chunk["qualified_name"] == "Calculator.add"
    )
    assert method_chunk["parent_class"] == "Calculator"
    assert method_chunk["symbol_name"] == "add"
    assert method_chunk["start_line"] is not None
    assert method_chunk["end_line"] is not None
    assert method_chunk["chunk_type"] == "code"


def test_build_python_code_chunks_falls_back_on_syntax_error():
    project_file = {
        "path": "app/broken.py",
        "name": "broken.py",
        "suffix": ".py",
        "content": "def broken(:\n    pass",
        "length": 20,
    }

    chunks = build_python_code_chunks(
        project_file=project_file,
        max_chunk_chars=100,
        overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0]["code_unit_type"] == "text_fallback"
    assert chunks[0]["parser"] == "text_fallback"
    assert chunks[0]["parse_error"]


def test_fallback_code_chunks_record_part_count():
    project_file = {
        "path": "app/broken_long.py",
        "name": "broken_long.py",
        "suffix": ".py",
        "content": "def broken(:\n" + "x = 1\n" * 30,
        "length": 200,
    }

    chunks = build_python_code_chunks(
        project_file=project_file,
        max_chunk_chars=30,
        overlap=5,
    )

    assert len(chunks) > 1
    assert all(chunk["part_count"] == len(chunks) for chunk in chunks)


def test_split_code_content_validates_params():
    with pytest.raises(ValueError):
        split_code_content("abc", max_chunk_chars=0, overlap=0)

    with pytest.raises(ValueError):
        split_code_content("abc", max_chunk_chars=10, overlap=-1)

    with pytest.raises(ValueError):
        split_code_content("abc", max_chunk_chars=10, overlap=10)


def test_long_function_is_split_into_parts():
    body = "\n".join(f"    value_{index} = {index}" for index in range(30))
    project_file = {
        "path": "app/long_func.py",
        "name": "long_func.py",
        "suffix": ".py",
        "content": f"def long_func():\n{body}\n    return value_1",
        "length": 500,
    }

    chunks = build_python_code_chunks(
        project_file=project_file,
        max_chunk_chars=80,
        overlap=10,
    )
    function_chunks = [
        chunk for chunk in chunks if chunk["qualified_name"] == "long_func"
    ]

    assert len(function_chunks) > 1
    assert all(chunk["part_count"] > 1 for chunk in function_chunks)
