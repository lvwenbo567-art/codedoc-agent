import sqlite3

import pytest

from day28_report_repository import (
    list_chunk_type_stats,
    list_file_chunk_counts,
    list_files_without_chunks,
    list_project_chunk_counts,
    list_project_file_counts,
    list_projects_with_chunk_count_greater_than,
)


def create_connection() -> sqlite3.Connection:
    """
    创建包含 projects、files、chunks 的内存数据库。
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
    conn.executemany(
        """
        INSERT INTO projects (
            project_path,
            file_count,
            chunk_count
        )
        VALUES (?, ?, ?)
        """,
        [
            ("project_a", 3, 3),
            ("project_b", 0, 0),
        ],
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
            (1, "api.py", "api.py", ".py", 100, "api"),
            (1, "README.md", "README.md", ".md", 80, "readme"),
            (1, "empty.py", "empty.py", ".py", 20, "empty"),
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
            (1, "api.py::0", "api.py", "api.py", ".py", "code", 0, "a", 10),
            (1, "api.py::1", "api.py", "api.py", ".py", "code", 1, "b", 30),
            (1, "README.md::0", "README.md", "README.md", ".md", "document", 0, "c", 20),
        ],
    )
    conn.commit()

    return conn


def test_list_project_file_counts_uses_left_join():
    conn = create_connection()

    results = list_project_file_counts(conn)

    assert results == [
        {
            "project_id": 1,
            "project_path": "project_a",
            "file_count": 3,
        },
        {
            "project_id": 2,
            "project_path": "project_b",
            "file_count": 0,
        },
    ]


def test_list_project_chunk_counts_uses_count_id_for_zero_match():
    conn = create_connection()

    results = list_project_chunk_counts(conn)

    project_b = [
        item
        for item in results
        if item["project_path"] == "project_b"
    ][0]

    assert project_b["chunk_count"] == 0


def test_list_chunk_type_stats_groups_by_type():
    conn = create_connection()

    results = list_chunk_type_stats(conn)
    code_stats = [
        item
        for item in results
        if item["chunk_type"] == "code"
    ][0]

    assert code_stats["chunk_count"] == 2
    assert code_stats["average_length"] == 20.0
    assert code_stats["maximum_length"] == 30
    assert code_stats["minimum_length"] == 10


def test_list_file_chunk_counts_counts_each_file():
    conn = create_connection()

    results = list_file_chunk_counts(conn)
    counts = {
        item["path"]: item["chunk_count"]
        for item in results
    }

    assert counts["api.py"] == 2
    assert counts["README.md"] == 1
    assert counts["empty.py"] == 0


def test_list_files_without_chunks():
    conn = create_connection()

    results = list_files_without_chunks(conn)

    assert results == [
        {
            "id": 3,
            "path": "empty.py",
        }
    ]


def test_list_projects_with_chunk_count_greater_than_uses_having():
    conn = create_connection()

    results = list_projects_with_chunk_count_greater_than(
        conn,
        min_chunk_count=2,
    )

    assert results == [
        {
            "project_id": 1,
            "project_path": "project_a",
            "chunk_count": 3,
        }
    ]


def test_list_projects_with_chunk_count_rejects_negative_threshold():
    conn = create_connection()

    with pytest.raises(ValueError):
        list_projects_with_chunk_count_greater_than(
            conn,
            min_chunk_count=-1,
        )
