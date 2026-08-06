from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from api_main import app


client = TestClient(app)


def test_scan_project_api_success(tmp_path):
    readme = tmp_path / "README.md"
    main_py = tmp_path / "main.py"
    config_json = tmp_path / "config.json"
    settings_yaml = tmp_path / "settings.yaml"

    readme.write_text("# Test Project\nThis is a test project.", encoding="utf-8")
    main_py.write_text("def main():\n    pass", encoding="utf-8")
    config_json.write_text('{"name": "codedoc"}', encoding="utf-8")
    settings_yaml.write_text("model: bge-m3\n", encoding="utf-8")

    response = client.post(
        "/scan",
        json={
            "project_path": str(tmp_path),
            "chunk_size": 100,
            "overlap": 20,
            "save_chunks": False,
            "output_path": "outputs/test_chunks.json",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["file_count"] == 4
    assert data["data"]["chunk_count"] == 4

    assert data["data"]["chunk_stats"]["total"] == 4
    assert data["data"]["chunk_stats"]["code_count"] == 1
    assert data["data"]["chunk_stats"]["document_count"] == 3

    assert len(data["data"]["files"]) == 4
    assert len(data["data"]["chunk_previews"]) == 3


def test_scan_project_api_path_not_exists():
    response = client.post(
        "/scan",
        json={
            "project_path": "not_exists_path",
            "chunk_size": 100,
            "overlap": 20,
            "save_chunks": False,
        },
    )

    assert response.status_code == 404


def test_scan_project_api_invalid_overlap(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Test Project", encoding="utf-8")

    response = client.post(
        "/scan",
        json={
            "project_path": str(tmp_path),
            "chunk_size": 100,
            "overlap": 100,
            "save_chunks": False,
        },
    )

    assert response.status_code == 400


def test_scan_project_api_save_chunks(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Test Project", encoding="utf-8")

    output_path = tmp_path / "chunks.json"

    response = client.post(
        "/scan",
        json={
            "project_path": str(tmp_path),
            "chunk_size": 100,
            "overlap": 20,
            "save_chunks": True,
            "output_path": str(output_path),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["data"]["saved_path"] == str(output_path)
    assert output_path.exists()
