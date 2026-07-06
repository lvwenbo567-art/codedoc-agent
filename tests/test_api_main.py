from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from api_main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "codedoc-agent"


def test_get_version():
    response = client.get("/version")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "codedoc-agent"
    assert data["version"] == "0.1.0"
    assert data["stage"] == "day13-fastapi-entry"


def test_get_config():
    response = client.get("/config")

    assert response.status_code == 200

    data = response.json()

    assert ".md" in data["supported_suffixes"]
    assert ".txt" in data["supported_suffixes"]
    assert ".py" in data["supported_suffixes"]

    assert data["default_chunk_size"] > 0
    assert data["default_chunk_overlap"] >= 0
    assert data["default_model_name"] == "mock-model"