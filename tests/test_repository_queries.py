from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from repositories.repository import (
    get_chunk_by_id,
    get_file_by_id,
    get_project_by_id,
    list_chunks,
    list_files,
    list_projects,
    save_project_snapshot,
)


def build_test_data():
    files = [
        {
            "path": "README.md",
            "name": "README.md",
            "suffix": ".md",
            "content": "# Test Project",
            "length": 14,
        },
        {
            "path": "main.py",
            "name": "main.py",
            "suffix": ".py",
            "content": "def main():\n    pass",
            "length": 20,
        },
    ]

    chunks = [
        {
            "chunk_id": "README.md::chunk_0",
            "source_path": "README.md",
            "source_name": "README.md",
            "source_suffix": ".md",
            "chunk_type": "document",
            "chunk_index": 0,
            "content": "# Test Project",
            "length": 14,
        },
        {
            "chunk_id": "main.py::chunk_0",
            "source_path": "main.py",
            "source_name": "main.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": 0,
            "content": "def main():\n    pass",
            "length": 20,
        },
    ]

    return files, chunks


def test_project_file_chunk_queries(tmp_path):
    db_path = str(tmp_path / "codedoc.db")

    files, chunks = build_test_data()

    project_id = save_project_snapshot(
        project_path="test_project",
        files=files,
        chunks=chunks,
        db_path=db_path,
    )

    project = get_project_by_id(
        project_id=project_id,
        db_path=db_path,
    )

    assert project is not None
    assert project["project_path"] == "test_project"
    assert project["file_count"] == 2
    assert project["chunk_count"] == 2

    projects = list_projects(
        db_path=db_path,
    )

    assert len(projects) == 1

    stored_files = list_files(
        project_id=project_id,
        db_path=db_path,
    )

    assert len(stored_files) == 2

    python_files = list_files(
        project_id=project_id,
        suffix=".py",
        db_path=db_path,
    )

    assert len(python_files) == 1
    assert python_files[0]["name"] == "main.py"

    file = get_file_by_id(
        file_id=python_files[0]["id"],
        db_path=db_path,
    )

    assert file is not None
    assert file["suffix"] == ".py"

    code_chunks = list_chunks(
        project_id=project_id,
        chunk_type="code",
        db_path=db_path,
    )

    assert len(code_chunks) == 1
    assert code_chunks[0]["source_name"] == "main.py"

    chunk = get_chunk_by_id(
        chunk_db_id=code_chunks[0]["id"],
        db_path=db_path,
    )

    assert chunk is not None
    assert chunk["content"] == "def main():\n    pass"


def test_repository_pagination(tmp_path):
    db_path = str(tmp_path / "codedoc.db")

    files, chunks = build_test_data()

    save_project_snapshot(
        project_path="project_1",
        files=files,
        chunks=chunks,
        db_path=db_path,
    )

    save_project_snapshot(
        project_path="project_2",
        files=files,
        chunks=chunks,
        db_path=db_path,
    )

    first_page = list_projects(
        limit=1,
        offset=0,
        db_path=db_path,
    )

    second_page = list_projects(
        limit=1,
        offset=1,
        db_path=db_path,
    )

    assert len(first_page) == 1
    assert len(second_page) == 1
    assert first_page[0]["id"] != second_page[0]["id"]


def test_invalid_pagination(tmp_path):
    db_path = str(tmp_path / "codedoc.db")

    with pytest.raises(ValueError):
        list_projects(
            limit=0,
            offset=0,
            db_path=db_path,
        )

    with pytest.raises(ValueError):
        list_files(
            limit=10,
            offset=-1,
            db_path=db_path,
        )


def test_save_project_snapshot_rolls_back_on_chunk_error(tmp_path):
    db_path = str(tmp_path / "codedoc.db")

    files, chunks = build_test_data()
    broken_chunks = [
        dict(chunks[0])
    ]
    broken_chunks[0].pop(
        "source_path"
    )

    with pytest.raises(KeyError):
        save_project_snapshot(
            project_path="broken_project",
            files=files,
            chunks=broken_chunks,
            db_path=db_path,
        )

    assert list_projects(
        db_path=db_path,
    ) == []
