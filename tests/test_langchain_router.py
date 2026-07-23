from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.langchain_router import router


app = FastAPI()
app.include_router(router)

client = TestClient(app)


def test_get_langchain_config(monkeypatch):
    monkeypatch.setenv(
        "LANGCHAIN_CHAT_PROVIDER",
        "mock",
    )

    response = client.get("/langchain/config")

    assert response.status_code == 200

    data = response.json()

    assert data["provider"] == "mock"
    assert "api_key" not in data


def test_mock_langchain_chat(monkeypatch):
    monkeypatch.setenv(
        "LANGCHAIN_CHAT_PROVIDER",
        "mock",
    )

    response = client.post(
        "/langchain/chat",
        json={
            "query": "测试 LangChain",
            "history": [],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "测试 LangChain" in data["answer"]


def test_mock_query_analysis(monkeypatch):
    monkeypatch.setenv(
        "LANGCHAIN_CHAT_PROVIDER",
        "mock",
    )

    response = client.post(
        "/langchain/analyze-query",
        json={
            "query": "RerankClient.score 在哪里实现？",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["analysis"]["query_type"] == "code"
    assert data["analysis"]["recommended_tool"] == "search_code"
