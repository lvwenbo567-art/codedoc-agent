import sqlite3

import pytest

from day27_sqlite_practice import (
    delete_chunk_by_id,
    insert_project,
    list_chunks_by_project,
    list_long_chunks,
    list_python_files_by_project,
    update_project_stats,
)


def create_connection() -> sqlite3.Connection:
    """
    创建内存 SQLite 数据库并写入测试数据。
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_path TEXT NOT NULL,
            file_count INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            path TEXT NOT NULL,
            name TEXT NOT NULL,
            suffix TEXT NOT NULL,
            length INTEGER NOT NULL,
            content_preview TEXT NOT NULL
        );

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            chunk_id TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_name TEXT NOT NULL,
            source_suffix TEXT NOT NULL,
            chunk_type TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            length INTEGER NOT NULL
        );
        """
    )
    project_id = insert_project(
        conn=conn,
        project_path="test_project",
        file_count=2,
        chunk_count=3,
    )
    conn.executemany(
        """
        INSERT INTO files (
            project_id,
            path,
            name,
            suffix,
            length,
            content_preview
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (project_id, "app.py", "app.py", ".py", 120, "def app"),
            (project_id, "README.md", "README.md", ".md", 80, "readme"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO chunks (
            project_id,
            chunk_id,
            source_path,
            source_name,
            source_suffix,
            chunk_type,
            chunk_index,
            content,
            length
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (project_id, "app.py::0", "app.py", "app.py", ".py", "code", 0, "a", 10),
            (project_id, "app.py::1", "app.py", "app.py", ".py", "code", 1, "b", 30),
            (project_id, "README.md::0", "README.md", "README.md", ".md", "document", 0, "c", 20),
        ],
    )
    conn.commit()

    return conn


def test_list_chunks_by_project_normal_query():
    conn = create_connection()

    results = list_chunks_by_project(conn, project_id=1)

    assert len(results) == 3
    assert results[0]["chunk_id"] == "app.py::0"


def test_list_chunks_by_project_no_match():
    conn = create_connection()

    assert list_chunks_by_project(conn, project_id=999) == []


def test_list_chunks_by_project_rejects_invalid_limit():
    conn = create_connection()

    with pytest.raises(ValueError):
        list_chunks_by_project(conn, project_id=1, limit=0)


def test_list_chunks_by_project_rejects_invalid_offset():
    conn = create_connection()

    with pytest.raises(ValueError):
        list_chunks_by_project(conn, project_id=1, offset=-1)


def test_pagination_pages_do_not_overlap():
    conn = create_connection()

    first_page = list_chunks_by_project(conn, project_id=1, limit=1, offset=0)
    second_page = list_chunks_by_project(conn, project_id=1, limit=1, offset=1)

    assert first_page[0]["id"] != second_page[0]["id"]


def test_list_python_files_by_project():
    conn = create_connection()

    results = list_python_files_by_project(conn, project_id=1)

    assert len(results) == 1
    assert results[0]["suffix"] == ".py"


def test_list_long_chunks_order_by_length_desc():
    conn = create_connection()

    results = list_long_chunks(conn, min_length=10, limit=3)

    assert [item["length"] for item in results] == [30, 20, 10]


def test_insert_and_update_project_stats():
    conn = create_connection()
    project_id = insert_project(conn, "another_project", 0, 0)

    update_project_stats(conn, project_id, file_count=5, chunk_count=8)
    row = conn.execute(
        "SELECT file_count, chunk_count FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()

    assert dict(row) == {
        "file_count": 5,
        "chunk_count": 8,
    }


def test_delete_chunk_by_id():
    conn = create_connection()

    delete_chunk_by_id(conn, chunk_db_id=1)

    row = conn.execute(
        "SELECT id FROM chunks WHERE id = ?",
        (1,),
    ).fetchone()

    assert row is None
