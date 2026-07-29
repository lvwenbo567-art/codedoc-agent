from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from api.langgraph_router import router
from langgraph_agent.workflow_service import CodeDocWorkflowExecutionError


def build_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_agentic_rag_empty_query_returns_422():
    response = build_client().post(
        "/langgraph/agentic-rag",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_agentic_rag_invalid_top_k_returns_422():
    response = build_client().post(
        "/langgraph/agentic-rag",
        json={
            "query": "hello",
            "candidate_top_k": 2,
            "final_top_k": 5,
        },
    )

    assert response.status_code == 422


def test_agentic_rag_unknown_question_returns_200():
    response = build_client().post(
        "/langgraph/agentic-rag",
        json={
            "query": "今天吃什么？",
            "project_root": ".",
            "chunks_path": "outputs/not-required.json",
            "index_path": "outputs/not-required.json",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query_type"] == "unknown"
    assert data["execution_steps"] == [
        "initialize",
        "analyze_query",
        "fallback_answer",
    ]


def test_agentic_rag_service_error_returns_500(monkeypatch):
    class BrokenService:
        async def arun(self, **kwargs):
            raise CodeDocWorkflowExecutionError("graph failed")

    def fake_builder(request):
        return BrokenService()

    monkeypatch.setattr(
        "api.langgraph_router.build_agentic_rag_service_for_request",
        fake_builder,
    )

    response = build_client().post(
        "/langgraph/agentic-rag",
        json={"query": "keyword_score 在哪里？"},
    )

    assert response.status_code == 500
