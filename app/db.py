import sqlite3
from pathlib import Path

from config import DEFAULT_DB_PATH


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    获取 SQLite 数据库连接。
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # SQLite 默认不会主动启用外键约束。
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    初始化数据库表。
    """
    conn = get_connection(db_path)

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_path TEXT NOT NULL,
                file_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                name TEXT NOT NULL,
                suffix TEXT NOT NULL,
                length INTEGER NOT NULL,
                content_preview TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                chunk_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_suffix TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                length INTEGER NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
            """
        )

        conn.commit()

    finally:
        conn.close()