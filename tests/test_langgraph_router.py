from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient


sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
        / "app"
    )
)

from api import langgraph_router as router_module


class FakeWorkflowService:
    async def arun(
        self,
        *,
        query: str,
        project_id: int,
    ) -> dict:
        return {
            "query": query,
            "project_id": project_id,
            "query_type": "code",
            "answer": "Fake graph answer",
            "evidence": [],
            "citations": [],
            "execution_steps": [
                "initialize",
                "classify_query",
                "code_search",
                "check_evidence",
                "build_answer",
            ],
            "evidence_sufficient": True,
            "error_message": None,
        }


def test_langgraph_workflow_api(monkeypatch):
    monkeypatch.setattr(
        router_module,
        "build_workflow_service_for_request",
        lambda request: FakeWorkflowService(),
    )

    app = FastAPI()
    app.include_router(router_module.router)
    client = TestClient(app)

    response = client.post(
        "/langgraph/workflow",
        json={
            "query": "RerankClient 在哪里定义？",
            "project_id": 1,
            "project_root": ".",
            "chunks_path": "outputs/chunks.json",
            "index_path": "outputs/vector_index.json",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query_type"] == "code"
    assert data["answer"] == "Fake graph answer"


def test_langgraph_request_validation():
    app = FastAPI()
    app.include_router(router_module.router)
    client = TestClient(app)

    response = client.post(
        "/langgraph/workflow",
        json={
            "query": "",
            "project_id": 0,
        },
    )

    assert response.status_code == 422
