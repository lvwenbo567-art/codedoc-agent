from __future__ import annotations

from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from api.persistent_agent_router import router


class FakePersistentService:
    async def arun(
        self,
        *,
        query: str,
        project_id: int,
        thread_id: str,
        recursion_limit: int,
    ):
        return {
            "query": query,
            "project_id": project_id,
            "thread_id": thread_id,
            "effective_thread_id": f"project:{project_id}:thread:{thread_id}",
            "run_id": "run_test",
            "answer": "ok",
            "success": True,
            "completed": True,
            "stop_reason": "completed",
            "turn_index": 1,
            "model_call_count": 1,
            "tool_call_count": 0,
            "message_count": 2,
            "message_trace": [],
            "tool_call_history": [],
            "execution_steps": ["initialize", "model_call", "finalize"],
            "checkpoint_id": "checkpoint-1",
            "total_duration_ms": 1.0,
            "error_message": None,
        }


class FakeInspectionCheckpointer:
    pass


class FakeRuntime:
    @property
    def checkpointer(self):
        return FakeInspectionCheckpointer()

    async def get_or_create_service(self, *, runtime, model_config):
        return FakePersistentService()

    def get_thread_lock(self, effective_thread_id: str):
        class FakeAsyncLock:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        return FakeAsyncLock()


@pytest.fixture
def client(monkeypatch) -> TestClient:
    from api import persistent_agent_router

    async def fake_latest_state(self, *, project_id: int, thread_id: str):
        return {
            "exists": True,
            "project_id": project_id,
            "thread_id": thread_id,
            "effective_thread_id": f"project:{project_id}:thread:{thread_id}",
            "checkpoint_id": "checkpoint-1",
            "created_at": "2026-07-29T00:00:00Z",
            "state": {"turn_index": 1},
        }

    async def fake_history(self, *, project_id: int, thread_id: str, limit: int):
        return [
            {
                "checkpoint_id": "checkpoint-1",
                "turn_index": 1,
            }
        ][:limit]

    async def fake_delete(self, *, project_id: int, thread_id: str):
        return {
            "deleted": True,
            "project_id": project_id,
            "thread_id": thread_id,
            "effective_thread_id": f"project:{project_id}:thread:{thread_id}",
        }

    monkeypatch.setattr(
        persistent_agent_router.CheckpointInspectionService,
        "get_latest_state",
        fake_latest_state,
    )
    monkeypatch.setattr(
        persistent_agent_router.CheckpointInspectionService,
        "list_history",
        fake_history,
    )
    monkeypatch.setattr(
        persistent_agent_router.CheckpointInspectionService,
        "delete_thread",
        fake_delete,
    )

    app = FastAPI()
    app.state.checkpoint_runtime = FakeRuntime()
    app.include_router(router)

    return TestClient(app)


def test_persistent_agent_success_returns_200(client: TestClient) -> None:
    response = client.post(
        "/langgraph/persistent-agent",
        json={
            "query": "hello",
            "project_id": 1,
            "thread_id": "day38-demo",
            "project_root": ".",
        },
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["effective_thread_id"] == "project:1:thread:day38-demo"


def test_persistent_agent_validation_errors(client: TestClient) -> None:
    assert client.post(
        "/langgraph/persistent-agent",
        json={"query": "", "project_id": 1, "thread_id": "x"},
    ).status_code == 422
    assert client.post(
        "/langgraph/persistent-agent",
        json={"query": "x", "project_id": 0, "thread_id": "x"},
    ).status_code == 422
    assert client.post(
        "/langgraph/persistent-agent",
        json={"query": "x", "project_id": 1, "thread_id": "bad/slash"},
    ).status_code == 400


def test_thread_state_history_and_delete_return_200(client: TestClient) -> None:
    state_response = client.get(
        "/langgraph/threads/day38-demo/state?project_id=1"
    )
    history_response = client.get(
        "/langgraph/threads/day38-demo/history?project_id=1&limit=20"
    )
    delete_response = client.delete(
        "/langgraph/threads/day38-demo?project_id=1"
    )

    assert state_response.status_code == 200
    assert state_response.json()["exists"] is True
    assert history_response.status_code == 200
    assert history_response.json()["count"] == 1
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True


def test_runtime_not_initialized_returns_500() -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/langgraph/persistent-agent",
        json={
            "query": "hello",
            "project_id": 1,
            "thread_id": "day38-demo",
            "project_root": ".",
        },
    )

    assert response.status_code == 500
