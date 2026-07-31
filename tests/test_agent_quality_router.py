from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

import api.agent_quality_router as agent_quality_router
from api_main import app
from repositories.feedback_repository import AgentFeedbackRepository
from services.feedback_service import AgentFeedbackService


def override_feedback_service(tmp_path: Path) -> None:
    repository = AgentFeedbackRepository(
        db_path=str(tmp_path / "feedback_api.db")
    )
    service = AgentFeedbackService(repository=repository)
    agent_quality_router.build_feedback_service = lambda: service


def feedback_payload() -> dict:
    return {
        "project_id": 1,
        "thread_id": "thread-1",
        "run_id": "run-1",
        "query": "keyword_score 在哪里？",
        "answer": "错误答案",
        "rating": -1,
        "issue_tags": ["incorrect_answer"],
        "comment": "定位错误",
        "corrected_answer": "keyword_score 在 search.py",
    }


def test_create_feedback_and_list_feedback(tmp_path: Path) -> None:
    override_feedback_service(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/agent-quality/feedback",
        json=feedback_payload(),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["feedback_id"] >= 1
    assert data["rating"] == -1

    list_response = client.get("/agent-quality/feedback?rating=-1")
    assert list_response.status_code == 200
    list_data = list_response.json()
    assert list_data["count"] == 1
    assert list_data["items"][0]["query"] == "keyword_score 在哪里？"


def test_create_feedback_invalid_rating_returns_422(
    tmp_path: Path,
) -> None:
    override_feedback_service(tmp_path)
    client = TestClient(app)
    payload = feedback_payload()
    payload["rating"] = 3

    response = client.post("/agent-quality/feedback", json=payload)

    assert response.status_code == 422


def test_promote_feedback_to_bad_case_and_list(tmp_path: Path) -> None:
    override_feedback_service(tmp_path)
    client = TestClient(app)
    feedback_response = client.post(
        "/agent-quality/feedback",
        json=feedback_payload(),
    )
    feedback_id = feedback_response.json()["feedback_id"]

    promote_response = client.post(
        f"/agent-quality/feedback/{feedback_id}/bad-case",
        json={
            "case_id": "bad-keyword-score",
            "name": "keyword_score 错误回答",
            "expected_tool_names": ["get_symbol_definition"],
            "forbidden_tool_names": [],
            "required_answer_terms": ["keyword_score", "search.py"],
            "accepted_stop_reasons": ["completed"],
            "notes": "用户反馈转化",
        },
    )

    assert promote_response.status_code == 200
    assert promote_response.json()["case_id"] == "bad-keyword-score"

    list_response = client.get("/agent-quality/bad-cases")
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1


def test_promote_missing_feedback_returns_404(tmp_path: Path) -> None:
    override_feedback_service(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/agent-quality/feedback/999/bad-case",
        json={
            "case_id": "bad-missing",
            "name": "Missing",
        },
    )

    assert response.status_code == 404
