from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

import api_main


def test_upload_project_zip_extracts_safe_archive(
    tmp_path,
) -> None:
    zip_path = tmp_path / "demo.zip"

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("demo/README.md", "# demo")
        archive.writestr("demo/main.py", "print('hello')")

    client = TestClient(api_main.app)

    response = client.post(
        "/project-upload/zip",
        data={
            "project_name": "demo project",
        },
        files={
            "file": (
                "demo.zip",
                zip_path.read_bytes(),
                "application/zip",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()
    data = body["data"]
    project_path = Path(data["project_path"])

    assert body["success"] is True
    assert data["project_name"] == "demo_project"
    assert data["extracted_file_count"] == 2
    assert project_path.name == "demo"
    assert (project_path / "README.md").exists()
    assert (project_path / "main.py").exists()
    assert Path(data["extract_root"]).name == "project"


def test_upload_project_zip_rejects_path_traversal(
    tmp_path,
) -> None:
    zip_path = tmp_path / "evil.zip"

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../evil.py", "print('bad')")

    client = TestClient(api_main.app)

    response = client.post(
        "/project-upload/zip",
        data={
            "project_name": "evil",
        },
        files={
            "file": (
                "evil.zip",
                zip_path.read_bytes(),
                "application/zip",
            )
        },
    )

    assert response.status_code == 400
    assert "非法路径" in response.json()["error"]["message"]
