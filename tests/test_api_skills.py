from fastapi.testclient import TestClient

from api_main import app


client = TestClient(app)


def test_list_skills_api():
    response = client.get("/skills")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    skill_names = {
        item["skill_name"]
        for item in data["data"]["skills"]
    }
    assert "project_onboarding" in skill_names
    assert "code_navigation" in skill_names
    assert "test_diagnosis" in skill_names


def test_route_skill_api():
    response = client.post(
        "/skills/route",
        json={
            "query": "这个项目有哪些主要目录和模块？"
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["data"]["route"]["skill_name"] == "project_onboarding"
    assert data["data"]["plan"]["recommended_tools"]
