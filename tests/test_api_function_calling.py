from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
        / "app"
    )
)


from fastapi.testclient import TestClient

from api_main import app


def test_function_calling_api_mock_structure(
    tmp_path,
):
    (
        tmp_path / "app"
    ).mkdir()
    (
        tmp_path
        / "app"
        / "main.py"
    ).write_text(
        "print('hello')",
        encoding="utf-8",
    )

    client = TestClient(app)

    response = client.post(
        "/agent/function-call",
        json={
            "query": (
                "这个项目有哪些主要目录和模块？"
            ),
            "project_root": str(tmp_path),
            "chunks_path": str(
                tmp_path / "chunks.json"
            ),
            "index_path": str(
                tmp_path / "index.json"
            ),
            "provider": "mock",
            "max_steps": 4,
        },
    )

    assert response.status_code == 200

    data = response.json()["data"]

    assert data["stop_reason"] == "final_answer"
    assert data["tool_call_count"] == 1
    assert (
        data["tool_traces"][0]["tool_name"]
        == "get_project_structure"
    )
    assert (
        data["tool_traces"][0]["result"]["success"]
        is True
    )


def test_function_calling_api_accepts_retrieval_config(
    tmp_path,
):
    (
        tmp_path / "app"
    ).mkdir()
    (
        tmp_path
        / "app"
        / "main.py"
    ).write_text(
        "print('hello')",
        encoding="utf-8",
    )

    client = TestClient(app)

    response = client.post(
        "/agent/function-call",
        json={
            "query": (
                "这个项目有哪些主要目录和模块？"
            ),
            "project_root": str(tmp_path),
            "chunks_path": str(
                tmp_path / "chunks.json"
            ),
            "index_path": str(
                tmp_path / "index.json"
            ),
            "provider": "mock",
            "embedding_provider": "ollama",
            "embedding_model": "bge-m3",
            "embedding_base_url": (
                "http://localhost:11434"
            ),
            "embedding_api_key": "",
            "dimension": 1024,
            "rerank_provider": (
                "sentence_transformers"
            ),
            "rerank_model": (
                "D:\\models\\bge-reranker-v2-m3"
            ),
            "rerank_device": "cpu",
            "max_steps": 4,
        },
    )

    assert response.status_code == 200

    data = response.json()["data"]

    assert data["tool_call_count"] == 1
    assert (
        data["tool_traces"][0]["tool_name"]
        == "get_project_structure"
    )
