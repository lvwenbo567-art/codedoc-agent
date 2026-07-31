from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_FEEDBACK_DB_PATH = "data/agent_feedback.db"


def utc_now_iso() -> str:
    """
    生成 UTC ISO 时间字符串。
    """
    return datetime.now(timezone.utc).isoformat()


class FeedbackNotFoundError(LookupError):
    """
    指定 feedback_id 不存在。
    """


class AgentFeedbackRepository:
    """
    负责 Feedback 和 Bad Case 的 SQLite 持久化。
    """

    def __init__(
        self,
        db_path: str = DEFAULT_FEEDBACK_DB_PATH,
    ) -> None:
        self.db_path = db_path
        self._ensure_database()

    def _connect(self) -> sqlite3.Connection:
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    thread_id TEXT NOT NULL,
                    run_id TEXT,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    issue_tags_json TEXT NOT NULL,
                    comment TEXT,
                    corrected_answer TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_bad_cases (
                    bad_case_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id INTEGER NOT NULL,
                    case_id TEXT NOT NULL UNIQUE,
                    project_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    query TEXT NOT NULL,
                    expected_tool_names_json TEXT NOT NULL,
                    forbidden_tool_names_json TEXT NOT NULL,
                    required_answer_terms_json TEXT NOT NULL,
                    accepted_stop_reasons_json TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(feedback_id)
                        REFERENCES agent_feedback(feedback_id)
                        ON DELETE CASCADE
                )
                """
            )

    @staticmethod
    def _row_to_feedback(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "feedback_id": row["feedback_id"],
            "project_id": row["project_id"],
            "thread_id": row["thread_id"],
            "run_id": row["run_id"],
            "query": row["query"],
            "answer": row["answer"],
            "rating": row["rating"],
            "issue_tags": json.loads(row["issue_tags_json"]),
            "comment": row["comment"],
            "corrected_answer": row["corrected_answer"],
            "created_at": row["created_at"],
        }

    @staticmethod
    def _row_to_bad_case(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "bad_case_id": row["bad_case_id"],
            "feedback_id": row["feedback_id"],
            "case_id": row["case_id"],
            "project_id": row["project_id"],
            "name": row["name"],
            "query": row["query"],
            "expected_tool_names": json.loads(
                row["expected_tool_names_json"]
            ),
            "forbidden_tool_names": json.loads(
                row["forbidden_tool_names_json"]
            ),
            "required_answer_terms": json.loads(
                row["required_answer_terms_json"]
            ),
            "accepted_stop_reasons": json.loads(
                row["accepted_stop_reasons_json"]
            ),
            "notes": row["notes"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_feedback(
        self,
        *,
        project_id: int,
        thread_id: str,
        run_id: str | None,
        query: str,
        answer: str,
        rating: int,
        issue_tags: list[str],
        comment: str | None,
        corrected_answer: str | None,
    ) -> dict[str, Any]:
        created_at = utc_now_iso()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO agent_feedback (
                    project_id,
                    thread_id,
                    run_id,
                    query,
                    answer,
                    rating,
                    issue_tags_json,
                    comment,
                    corrected_answer,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    thread_id,
                    run_id,
                    query,
                    answer,
                    rating,
                    json.dumps(issue_tags, ensure_ascii=False),
                    comment,
                    corrected_answer,
                    created_at,
                ),
            )
            feedback_id = int(cursor.lastrowid)

        feedback = self.get_feedback(feedback_id)

        if feedback is None:
            raise RuntimeError("Feedback 创建后读取失败")

        return feedback

    def get_feedback(self, feedback_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM agent_feedback
                WHERE feedback_id = ?
                """,
                (feedback_id,),
            ).fetchone()

        return self._row_to_feedback(row) if row else None

    def list_feedback(
        self,
        *,
        project_id: int | None = None,
        rating: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []

        if project_id is not None:
            conditions.append("project_id = ?")
            params.append(project_id)

        if rating is not None:
            conditions.append("rating = ?")
            params.append(rating)

        where_sql = (
            "WHERE " + " AND ".join(conditions) if conditions else ""
        )

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM agent_feedback
                {where_sql}
                ORDER BY feedback_id DESC
                LIMIT ?
                OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()

        return [self._row_to_feedback(row) for row in rows]

    def delete_feedback(self, feedback_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM agent_feedback
                WHERE feedback_id = ?
                """,
                (feedback_id,),
            )

        return cursor.rowcount > 0

    def promote_feedback_to_bad_case(
        self,
        *,
        feedback_id: int,
        case_id: str,
        name: str,
        expected_tool_names: list[str],
        forbidden_tool_names: list[str],
        required_answer_terms: list[str],
        accepted_stop_reasons: list[str],
        notes: str | None,
    ) -> dict[str, Any]:
        feedback = self.get_feedback(feedback_id)

        if feedback is None:
            raise FeedbackNotFoundError(
                f"Feedback 不存在：{feedback_id}"
            )

        now = utc_now_iso()

        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT created_at
                FROM agent_bad_cases
                WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()
            created_at = existing["created_at"] if existing else now

            connection.execute(
                """
                INSERT INTO agent_bad_cases (
                    feedback_id,
                    case_id,
                    project_id,
                    name,
                    query,
                    expected_tool_names_json,
                    forbidden_tool_names_json,
                    required_answer_terms_json,
                    accepted_stop_reasons_json,
                    notes,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    feedback_id = excluded.feedback_id,
                    project_id = excluded.project_id,
                    name = excluded.name,
                    query = excluded.query,
                    expected_tool_names_json = excluded.expected_tool_names_json,
                    forbidden_tool_names_json = excluded.forbidden_tool_names_json,
                    required_answer_terms_json = excluded.required_answer_terms_json,
                    accepted_stop_reasons_json = excluded.accepted_stop_reasons_json,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (
                    feedback_id,
                    case_id,
                    feedback["project_id"],
                    name,
                    feedback["query"],
                    json.dumps(expected_tool_names, ensure_ascii=False),
                    json.dumps(forbidden_tool_names, ensure_ascii=False),
                    json.dumps(required_answer_terms, ensure_ascii=False),
                    json.dumps(accepted_stop_reasons, ensure_ascii=False),
                    notes,
                    created_at,
                    now,
                ),
            )

            row = connection.execute(
                """
                SELECT *
                FROM agent_bad_cases
                WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()

        return self._row_to_bad_case(row)

    def list_bad_cases(
        self,
        *,
        project_id: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []

        if project_id is not None:
            conditions.append("project_id = ?")
            params.append(project_id)

        where_sql = (
            "WHERE " + " AND ".join(conditions) if conditions else ""
        )

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM agent_bad_cases
                {where_sql}
                ORDER BY bad_case_id DESC
                LIMIT ?
                OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()

        return [self._row_to_bad_case(row) for row in rows]
