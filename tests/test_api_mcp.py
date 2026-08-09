from fastapi.testclient import TestClient

from api_main import app


client = TestClient(app)


def test_list_mcp_tools_api():
    response = client.get("/mcp/tools")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    names = {
        item["name"]
        for item in data["data"]["tools"]
    }
    assert "get_project_structure" in names
    assert "run_project_tests" not in names


def test_call_mcp_tool_api():
    response = client.post(
        "/mcp/tools/call",
        json={
            "tool_name": "get_project_structure",
            "project_root": "test_project",
            "arguments": {
                "max_depth": 1,
                "max_entries": 20,
                "include_files": True,
                "include_hidden": False,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["data"]["success"] is True
    assert data["data"]["data"]["entry_count"] > 0


def test_list_mcp_prompts_and_resources_api():
    prompts = client.get("/mcp/prompts")
    resources = client.get("/mcp/resources?project_id=1")

    assert prompts.status_code == 200
    assert resources.status_code == 200

    assert prompts.json()["data"]["prompts"]
    assert resources.json()["data"]["resources"]
