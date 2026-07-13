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

    assert data["success"] is True
    assert data["data"]["status"] == "ok"
    assert data["data"]["service"] == "codedoc-agent"


def test_get_version():
    response = client.get("/version")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["service"] == "codedoc-agent"
    assert data["data"]["version"] == "0.1.0"
    assert data["data"]["stage"] == "day21-rag-ask"


def test_get_config():
    response = client.get("/config")

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    config = data["data"]

    assert ".md" in config["supported_suffixes"]
    assert ".txt" in config["supported_suffixes"]
    assert ".py" in config["supported_suffixes"]

    assert config["default_chunk_size"] > 0
    assert config["default_chunk_overlap"] >= 0
    assert config["default_model_name"] == "mock-model"
