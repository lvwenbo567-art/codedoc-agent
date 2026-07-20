import sqlite3
from typing import Any


def validate_positive_int(value: int, name: str) -> None:
    """
    校验参数必须是正整数。
    """
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")


def validate_non_negative_int(value: int, name: str) -> None:
    """
    校验参数不能是负数。
    """
    if value < 0:
        raise ValueError(f"{name} 不能小于 0")


def fetch_all_as_dicts(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    """
    将 sqlite3.Row 查询结果转换成普通字典列表。
    """
    return [
        dict(row)
        for row in cursor.fetchall()
    ]


def list_chunks_by_project(
    conn: sqlite3.Connection,
    project_id: int,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    使用参数化 SQL 分页查询指定项目下的 chunks。
    """
    validate_positive_int(project_id, "project_id")
    validate_positive_int(limit, "limit")
    validate_non_negative_int(offset, "offset")
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT
            id,
            project_id,
            chunk_id,
            source_path,
            chunk_type,
            length
        FROM chunks
        WHERE project_id = ?
        ORDER BY id ASC
        LIMIT ? OFFSET ?
        """,
        (
            project_id,
            limit,
            offset,
        ),
    )

    return fetch_all_as_dicts(cursor)


def list_python_files_by_project(
    conn: sqlite3.Connection,
    project_id: int,
) -> list[dict[str, Any]]:
    """
    使用参数化 SQL 查询指定项目下的 Python 文件。
    """
    validate_positive_int(project_id, "project_id")
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT
            id,
            project_id,
            path,
            name,
            suffix,
            length
        FROM files
        WHERE project_id = ?
          AND suffix = ?
        ORDER BY id ASC
        """,
        (
            project_id,
            ".py",
        ),
    )

    return fetch_all_as_dicts(cursor)


def list_long_chunks(
    conn: sqlite3.Connection,
    min_length: int,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    使用参数化 SQL 查询长度大于等于指定值的 chunks。
    """
    validate_non_negative_int(min_length, "min_length")
    validate_positive_int(limit, "limit")
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT
            id,
            chunk_id,
            chunk_type,
            length
        FROM chunks
        WHERE length >= ?
        ORDER BY length DESC, id ASC
        LIMIT ?
        """,
        (
            min_length,
            limit,
        ),
    )

    return fetch_all_as_dicts(cursor)


def insert_project(
    conn: sqlite3.Connection,
    project_path: str,
    file_count: int,
    chunk_count: int,
) -> int:
    """
    使用参数化 SQL 插入项目记录，并返回新项目 ID。
    """
    if not project_path or not project_path.strip():
        raise ValueError("project_path 不能为空")

    validate_non_negative_int(file_count, "file_count")
    validate_non_negative_int(chunk_count, "chunk_count")
    cursor = conn.execute(
        """
        INSERT INTO projects (
            project_path,
            file_count,
            chunk_count
        )
        VALUES (?, ?, ?)
        """,
        (
            project_path,
            file_count,
            chunk_count,
        ),
    )
    conn.commit()

    return int(cursor.lastrowid)


def update_project_stats(
    conn: sqlite3.Connection,
    project_id: int,
    file_count: int,
    chunk_count: int,
) -> None:
    """
    使用参数化 SQL 更新项目文件数和 chunk 数。
    """
    validate_positive_int(project_id, "project_id")
    validate_non_negative_int(file_count, "file_count")
    validate_non_negative_int(chunk_count, "chunk_count")
    conn.execute(
        """
        UPDATE projects
        SET
            file_count = ?,
            chunk_count = ?
        WHERE id = ?
        """,
        (
            file_count,
            chunk_count,
            project_id,
        ),
    )
    conn.commit()


def delete_chunk_by_id(
    conn: sqlite3.Connection,
    chunk_db_id: int,
) -> None:
    """
    使用参数化 SQL 删除指定数据库 ID 的 chunk。
    """
    validate_positive_int(chunk_db_id, "chunk_db_id")
    conn.execute(
        """
        DELETE FROM chunks
        WHERE id = ?
        """,
        (chunk_db_id,),
    )
    conn.commit()
