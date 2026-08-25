from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
        / "app"
    )
)


import tools.code_doc_tools as code_doc_tools

from tools.code_doc_tools import (
    build_code_doc_tool_registry,
)
from tools.executor import ToolExecutor


def test_project_structure_tool(
    tmp_path,
):
    app_directory = (
        tmp_path / "app"
    )

    app_directory.mkdir()

    (
        app_directory
        / "main.py"
    ).write_text(
        "print('hello')",
        encoding="utf-8",
    )

    (
        tmp_path
        / "README.md"
    ).write_text(
        "# Demo",
        encoding="utf-8",
    )

    registry = (
        build_code_doc_tool_registry(
            project_root=str(
                tmp_path
            ),
            chunks_path=(
                str(
                    tmp_path
                    / "chunks.json"
                )
            ),
            index_path=(
                str(
                    tmp_path
                    / "index.json"
                )
            ),
        )
    )

    executor = ToolExecutor(
        registry
    )

    result = executor.execute(
        tool_name=(
            "get_project_structure"
        ),
        arguments={
            "max_depth": 3,
            "max_entries": 100,
            "include_files": True,
            "include_hidden": False,
        },
    )

    assert result.success is True

    paths = {
        item["path"]
        for item
        in result.data["entries"]
    }

    assert "app" in paths
    assert "app/main.py" in paths
    assert "README.md" in paths


def test_project_structure_lists_top_level_directories_before_deep_files(
    tmp_path,
):
    """大目录不能挤掉同级的 docs/tests 等主要模块。"""
    for directory_name in ("app", "docs", "tests"):
        (tmp_path / directory_name).mkdir()

    for index in range(10):
        (tmp_path / "app" / f"module_{index}.py").write_text(
            "pass\n",
            encoding="utf-8",
        )

    registry = build_code_doc_tool_registry(
        project_root=str(tmp_path),
        chunks_path=str(tmp_path / "chunks.json"),
        index_path=str(tmp_path / "index.json"),
    )
    result = ToolExecutor(registry).execute(
        tool_name="get_project_structure",
        arguments={
            "max_depth": 3,
            "max_entries": 10,
            "include_files": True,
            "include_hidden": False,
        },
    )

    paths = [item["path"] for item in result.data["entries"]]
    assert paths[:3] == ["app", "docs", "tests"]


def test_search_code_uses_code_filter(
    tmp_path,
    monkeypatch,
):
    chunks_path = (
        tmp_path / "chunks.json"
    )

    index_path = (
        tmp_path / "index.json"
    )

    chunks_path.write_text(
        "[]",
        encoding="utf-8",
    )

    index_path.write_text(
        "{}",
        encoding="utf-8",
    )

    captured: dict = {}

    def fake_retrieve_with_rerank(
        **kwargs,
    ):
        captured.update(kwargs)

        return {
            "retrieval_mode": (
                "hybrid_rerank"
            ),
            "degraded": False,
            "candidate_count": 1,
            "results": [
                {
                    "rank": 1,
                    "chunk_id": "chunk-1",
                    "source_path": (
                        "app/example.py"
                    ),
                    "chunk_type": "code",
                    "qualified_name": (
                        "Example.run"
                    ),
                    "start_line": 10,
                    "end_line": 20,
                    "rerank_score": 0.9,
                    "content": (
                        "def run(self): ..."
                    ),
                }
            ],
        }

    monkeypatch.setattr(
        code_doc_tools,
        "retrieve_with_rerank",
        fake_retrieve_with_rerank,
    )

    registry = (
        build_code_doc_tool_registry(
            project_root=str(
                tmp_path
            ),
            chunks_path=str(
                chunks_path
            ),
            index_path=str(
                index_path
            ),
            embedding_provider="ollama",
            embedding_model="bge-m3",
            embedding_base_url="http://localhost:11434",
            embedding_api_key="",
            mock_dimension=1024,
            rerank_provider="sentence_transformers",
            rerank_model="D:\\models\\bge-reranker-v2-m3",
        )
    )

    executor = ToolExecutor(
        registry
    )

    result = executor.execute(
        tool_name="search_code",
        arguments={
            "query": (
                "Example.run 在哪里实现"
            ),
            "top_k": 3,
            "candidate_top_k": 10,
            "query_strategy": (
                "original"
            ),
        },
    )

    assert result.success is True

    assert (
        captured["chunk_type"]
        == "code"
    )

    assert (
        captured["final_top_k"]
        == 3
    )

    assert (
        captured["embedding_provider"]
        == "ollama"
    )

    assert (
        captured["embedding_model"]
        == "bge-m3"
    )

    assert (
        captured["mock_dimension"]
        == 1024
    )

    assert (
        captured["rerank_provider"]
        == "sentence_transformers"
    )

    assert (
        result.data["result_count"]
        == 1
    )


def test_search_documents_uses_document_filter(
    tmp_path,
    monkeypatch,
):
    chunks_path = (
        tmp_path / "chunks.json"
    )

    index_path = (
        tmp_path / "index.json"
    )

    chunks_path.write_text(
        "[]",
        encoding="utf-8",
    )

    index_path.write_text(
        "{}",
        encoding="utf-8",
    )

    captured: dict = {}

    def fake_retrieve_with_rerank(
        **kwargs,
    ):
        captured.update(kwargs)

        return {
            "results": [],
            "candidate_count": 0,
            "degraded": False,
        }

    monkeypatch.setattr(
        code_doc_tools,
        "retrieve_with_rerank",
        fake_retrieve_with_rerank,
    )

    registry = (
        build_code_doc_tool_registry(
            project_root=str(
                tmp_path
            ),
            chunks_path=str(
                chunks_path
            ),
            index_path=str(
                index_path
            ),
        )
    )

    executor = ToolExecutor(
        registry
    )

    result = executor.execute(
        tool_name="search_documents",
        arguments={
            "query": "项目如何启动",
            "top_k": 5,
            "candidate_top_k": 20,
            "query_strategy": (
                "multi_query"
            ),
        },
    )

    assert result.success is True

    assert (
        captured["chunk_type"]
        == "document"
    )
