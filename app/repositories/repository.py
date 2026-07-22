from typing import Dict, List, Optional

from config import DEFAULT_DB_PATH
from repositories.db import get_connection, init_db


def row_to_dict(row) -> Dict:
    """
    将 sqlite3.Row 转换为普通字典。
    """
    return dict(row)


def validate_pagination(limit: int, offset: int) -> None:
    """
    校验分页参数。
    """
    if limit <= 0:
        raise ValueError("limit 必须大于 0")

    if offset < 0:
        raise ValueError("offset 不能小于 0")


def save_project_snapshot(
    project_path: str,
    files: List[Dict],
    chunks: List[Dict],
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """
    保存一次项目扫描结果。

    返回新创建的 project_id。
    """
    init_db(db_path)

    conn = get_connection(db_path)

    try:
        cursor = conn.cursor()

        cursor.execute(
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
                len(files),
                len(chunks),
            ),
        )

        project_id = cursor.lastrowid

        for file in files:
            cursor.execute(
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
                (
                    project_id,
                    file["path"],
                    file["name"],
                    file["suffix"],
                    file["length"],
                    file["content"][:300],
                ),
            )

        for chunk in chunks:
            cursor.execute(
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
                (
                    project_id,
                    chunk["chunk_id"],
                    chunk["source_path"],
                    chunk["source_name"],
                    chunk["source_suffix"],
                    chunk["chunk_type"],
                    chunk["chunk_index"],
                    chunk["content"],
                    chunk["length"],
                ),
            )

        conn.commit()

        return project_id

    except Exception:
        # 保存 projects、files、chunks 是一个整体操作。
        # 任何一步失败，都回滚前面已经执行的写入。
        conn.rollback()
        raise

    finally:
        conn.close()


def list_projects(
    limit: int = 50,
    offset: int = 0,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict]:
    """
    分页查询项目扫描记录。
    """
    init_db(db_path)
    validate_pagination(limit, offset)

    conn = get_connection(db_path)

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                project_path,
                file_count,
                chunk_count,
                created_at
            FROM projects
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (
                limit,
                offset,
            ),
        )

        rows = cursor.fetchall()

        return [row_to_dict(row) for row in rows]

    finally:
        conn.close()


def get_project_by_id(
    project_id: int,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Dict]:
    """
    根据 project_id 查询项目详情。
    """
    init_db(db_path)

    conn = get_connection(db_path)

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                project_path,
                file_count,
                chunk_count,
                created_at
            FROM projects
            WHERE id = ?
            LIMIT 1
            """,
            (project_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return row_to_dict(row)

    finally:
        conn.close()


def list_files(
    project_id: Optional[int] = None,
    suffix: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict]:
    """
    查询文件记录。

    可以按 project_id 和 suffix 过滤。
    """
    init_db(db_path)
    validate_pagination(limit, offset)

    conn = get_connection(db_path)

    try:
        cursor = conn.cursor()

        sql = """
            SELECT
                id,
                project_id,
                path,
                name,
                suffix,
                length,
                content_preview
            FROM files
            WHERE 1 = 1
        """

        params = []

        if project_id is not None:
            sql += " AND project_id = ?"
            params.append(project_id)

        if suffix is not None:
            sql += " AND suffix = ?"
            params.append(suffix.lower())

        sql += " ORDER BY id ASC LIMIT ? OFFSET ?"

        params.append(limit)
        params.append(offset)

        cursor.execute(sql, params)

        rows = cursor.fetchall()

        return [row_to_dict(row) for row in rows]

    finally:
        conn.close()


def get_file_by_id(
    file_id: int,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Dict]:
    """
    根据数据库 ID 查询文件记录。
    """
    init_db(db_path)

    conn = get_connection(db_path)

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                project_id,
                path,
                name,
                suffix,
                length,
                content_preview
            FROM files
            WHERE id = ?
            LIMIT 1
            """,
            (file_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return row_to_dict(row)

    finally:
        conn.close()


def list_chunks(
    project_id: Optional[int] = None,
    chunk_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict]:
    """
    查询 chunks。

    可以按 project_id 和 chunk_type 过滤。
    """
    init_db(db_path)
    validate_pagination(limit, offset)

    conn = get_connection(db_path)

    try:
        cursor = conn.cursor()

        sql = """
            SELECT
                id,
                project_id,
                chunk_id,
                source_path,
                source_name,
                source_suffix,
                chunk_type,
                chunk_index,
                substr(content, 1, 300) AS content_preview,
                length
            FROM chunks
            WHERE 1 = 1
        """

        params = []

        if project_id is not None:
            sql += " AND project_id = ?"
            params.append(project_id)

        if chunk_type is not None:
            sql += " AND chunk_type = ?"
            params.append(chunk_type)

        sql += " ORDER BY id ASC LIMIT ? OFFSET ?"

        params.append(limit)
        params.append(offset)

        cursor.execute(sql, params)

        rows = cursor.fetchall()

        return [row_to_dict(row) for row in rows]

    finally:
        conn.close()


def get_chunk_by_id(
    chunk_db_id: int,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Dict]:
    """
    根据数据库自增 ID 查询单个 chunk。
    """
    init_db(db_path)

    conn = get_connection(db_path)

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                project_id,
                chunk_id,
                source_path,
                source_name,
                source_suffix,
                chunk_type,
                chunk_index,
                content,
                length
            FROM chunks
            WHERE id = ?
            LIMIT 1
            """,
            (chunk_db_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return row_to_dict(row)

    finally:
        conn.close()