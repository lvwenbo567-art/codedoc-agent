from __future__ import annotations

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import tool_agent_router
from api.tool_agent_router import router
from langgraph_agent.tool_agent_service import CodeDocToolAgentExecutionError


class FakeService:
    async def arun(self, *, query: str, project_id: int, recursion_limit: int):
        return {
            "query": query,
            "project_id": project_id,
            "answer": "ok",
            "success": True,
            "completed": True,
            "stop_reason": "completed",
            "error_message": None,
            "model_call_count": 1,
            "tool_call_count": 0,
            "tool_call_history": [],
            "execution_steps": ["initialize", "model_call", "finalize"],
            "message_count": 2,
            "message_trace": [],
            "allowed_tools": ["search_code"],
            "provider": "mock",
            "model_name": "fake",
        }


class ErrorService:
    async def arun(self, *, query: str, project_id: int, recursion_limit: int):
        raise CodeDocToolAgentExecutionError("boom")


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_router_success_returns_200(monkeypatch, client: TestClient) -> None:
    monkeypatch.setattr(
        tool_agent_router,
        "build_tool_agent_service",
        lambda request: FakeService(),
    )

    response = client.post(
        "/langgraph/tool-agent",
        json={"query": "hello", "project_root": "."},
    )

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_router_validation_errors(client: TestClient) -> None:
    assert client.post(
        "/langgraph/tool-agent",
        json={"query": "", "project_root": "."},
    ).status_code == 422
    assert client.post(
        "/langgraph/tool-agent",
        json={"query": "x", "project_id": 0, "project_root": "."},
    ).status_code == 422
    assert client.post(
        "/langgraph/tool-agent",
        json={"query": "x", "max_model_calls": 0, "project_root": "."},
    ).status_code == 422
    assert client.post(
        "/langgraph/tool-agent",
        json={"query": "x", "recursion_limit": 1, "project_root": "."},
    ).status_code == 422


def test_router_mock_provider_returns_400(monkeypatch, client: TestClient) -> None:
    monkeypatch.setenv("LANGCHAIN_CHAT_PROVIDER", "mock")

    response = client.post(
        "/langgraph/tool-agent",
        json={"query": "hello", "project_root": "."},
    )

    assert response.status_code == 400


def test_router_service_exception_returns_500(monkeypatch, client: TestClient) -> None:
    monkeypatch.setattr(
        tool_agent_router,
        "build_tool_agent_service",
        lambda request: ErrorService(),
    )

    response = client.post(
        "/langgraph/tool-agent",
        json={"query": "hello", "project_root": "."},
    )

    assert response.status_code == 500
