from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from api import human_review_router


class FakeService:
    async def start(self, **kwargs):
        return _response(query=kwargs["query"])

    async def resume(self, **kwargs):
        return _response(query="")


def _response(query: str) -> dict:
    return {
        "query": query,
        "project_id": 1,
        "thread_id": "t1",
        "effective_thread_id": "project:1:thread:t1",
        "run_id": "run_1",
        "answer": "",
        "status": "interrupted",
        "success": False,
        "completed": False,
        "stop_reason": "interrupted",
        "interrupts": [{"type": "tool_approval"}],
        "approval_status": "pending",
        "review_history": [],
        "turn_index": 1,
        "model_call_count": 1,
        "tool_call_count": 0,
        "message_count": 2,
        "message_trace": [],
        "tool_call_history": [],
        "execution_steps": ["controller_review"],
        "checkpoint_id": "checkpoint_1",
        "total_duration_ms": 1.0,
        "error_message": None,
        "allowed_tools": ["read_file_range"],
        "provider": "mock",
        "model_name": "fake",
    }


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(human_review_router.router)
    return app


def test_start_returns_200(monkeypatch) -> None:
    async def fake_get_hitl_service(*, body, request):
        return FakeService()

    monkeypatch.setattr(
        human_review_router,
        "get_hitl_service",
        fake_get_hitl_service,
    )
    client = TestClient(_app())
    response = client.post(
        "/langgraph/hitl/start",
        json={
            "query": "读代码",
            "project_id": 1,
            "thread_id": "t1",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "interrupted"


def test_resume_returns_200(monkeypatch) -> None:
    async def fake_get_hitl_service(*, body, request):
        return FakeService()

    monkeypatch.setattr(
        human_review_router,
        "get_hitl_service",
        fake_get_hitl_service,
    )
    client = TestClient(_app())
    response = client.post(
        "/langgraph/hitl/resume",
        json={
            "project_id": 1,
            "thread_id": "t1",
            "decision": {"decision": "approve"},
        },
    )

    assert response.status_code == 200


def test_invalid_decision_returns_422() -> None:
    client = TestClient(_app())
    response = client.post(
        "/langgraph/hitl/resume",
        json={
            "project_id": 1,
            "thread_id": "t1",
            "decision": {"decision": "edit"},
        },
    )

    assert response.status_code == 422


def test_stream_content_type(monkeypatch) -> None:
    class FakeSSEService:
        def __init__(self, *, agent_service):
            pass

        async def stream_start(self, **kwargs):
            yield "event: connected\ndata: {}\n\n"

    async def fake_get_hitl_service(*, body, request):
        return FakeService()

    monkeypatch.setattr(
        human_review_router,
        "get_hitl_service",
        fake_get_hitl_service,
    )
    monkeypatch.setattr(
        human_review_router,
        "HumanReviewSSEService",
        FakeSSEService,
    )
    client = TestClient(_app())
    response = client.post(
        "/langgraph/hitl/stream",
        json={
            "query": "读代码",
            "project_id": 1,
            "thread_id": "t1",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_runtime_not_started_returns_error() -> None:
    client = TestClient(_app())
    response = client.post(
        "/langgraph/hitl/start",
        json={
            "query": "读代码",
            "project_id": 1,
            "thread_id": "t1",
        },
    )

    assert response.status_code == 400
