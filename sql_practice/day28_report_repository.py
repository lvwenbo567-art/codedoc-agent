import sqlite3
from typing import Any


def fetch_all_as_dicts(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    """
    将 SQLite 查询结果转换为普通字典列表。
    """
    return [
        dict(row)
        for row in cursor.fetchall()
    ]


def list_project_file_counts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """
    查询每个项目的文件数量，使用 LEFT JOIN 保留无文件项目。
    """
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT
            p.id AS project_id,
            p.project_path,
            COUNT(f.id) AS file_count
        FROM projects AS p
        LEFT JOIN files AS f
            ON f.project_id = p.id
        GROUP BY
            p.id,
            p.project_path
        ORDER BY p.id ASC
        """
    )

    return fetch_all_as_dicts(cursor)


def list_project_chunk_counts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """
    查询每个项目的 chunk 数量，使用 COUNT(c.id) 避免误数空行。
    """
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT
            p.id AS project_id,
            p.project_path,
            COUNT(c.id) AS chunk_count
        FROM projects AS p
        LEFT JOIN chunks AS c
            ON c.project_id = p.id
        GROUP BY
            p.id,
            p.project_path
        ORDER BY chunk_count DESC, p.id ASC
        """
    )

    return fetch_all_as_dicts(cursor)


def list_chunk_type_stats(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """
    按 chunk_type 分组统计数量、平均长度、最大长度和最小长度。
    """
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT
            chunk_type,
            COUNT(*) AS chunk_count,
            ROUND(AVG(length), 2) AS average_length,
            MAX(length) AS maximum_length,
            MIN(length) AS minimum_length
        FROM chunks
        GROUP BY chunk_type
        ORDER BY chunk_count DESC, chunk_type ASC
        """
    )

    return fetch_all_as_dicts(cursor)


def list_file_chunk_counts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """
    查询每个文件对应的 chunk 数量。
    """
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT
            f.id AS file_id,
            f.path,
            COUNT(c.id) AS chunk_count
        FROM files AS f
        LEFT JOIN chunks AS c
            ON c.project_id = f.project_id
           AND c.source_path = f.path
        GROUP BY
            f.id,
            f.path
        ORDER BY chunk_count DESC, f.id ASC
        """
    )

    return fetch_all_as_dicts(cursor)


def list_files_without_chunks(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """
    查询 files 表中存在、但 chunks 表中没有对应记录的文件。
    """
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT
            f.id,
            f.path
        FROM files AS f
        LEFT JOIN chunks AS c
            ON c.project_id = f.project_id
           AND c.source_path = f.path
        WHERE c.id IS NULL
        ORDER BY f.id ASC
        """
    )

    return fetch_all_as_dicts(cursor)


def list_projects_with_chunk_count_greater_than(
    conn: sqlite3.Connection,
    min_chunk_count: int,
) -> list[dict[str, Any]]:
    """
    使用 GROUP BY + HAVING 查询 chunk 数量大于指定值的项目。
    """
    if min_chunk_count < 0:
        raise ValueError("min_chunk_count 不能小于 0")

    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT
            p.id AS project_id,
            p.project_path,
            COUNT(c.id) AS chunk_count
        FROM projects AS p
        INNER JOIN chunks AS c
            ON c.project_id = p.id
        GROUP BY
            p.id,
            p.project_path
        HAVING COUNT(c.id) > ?
        ORDER BY chunk_count DESC, p.id ASC
        """,
        (min_chunk_count,),
    )

    return fetch_all_as_dicts(cursor)
