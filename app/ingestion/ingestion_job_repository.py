from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiosqlite

from ingestion.ingestion_job_models import IngestionJobRecord, IngestionJobStatus, IngestionStage, utc_now_iso


class IngestionJobNotFoundError(ValueError):
    pass


class IngestionJobRepository:
    """SQLite 中保存单机 Ingestion Job 的状态机。"""

    def __init__(self, *, database_path: str) -> None:
        self.database_path = str(Path(database_path).resolve())
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("Job Repository 尚未启动")
        return self._connection

    async def start(self) -> None:
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.database_path)
        self._connection.row_factory = aiosqlite.Row
        await self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            job_id TEXT PRIMARY KEY, project_id INTEGER NOT NULL, status TEXT NOT NULL,
            stage TEXT NOT NULL, progress REAL NOT NULL, attempt INTEGER NOT NULL,
            request_json TEXT NOT NULL, result_json TEXT NOT NULL DEFAULT '{}',
            parent_job_id TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL, started_at TEXT,
            finished_at TEXT, error_type TEXT, error_message TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_project ON ingestion_jobs(project_id);
        CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs(status);
        """)
        await self.connection.commit()

    async def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()

    @staticmethod
    def _record(row: aiosqlite.Row) -> IngestionJobRecord:
        return IngestionJobRecord(
            job_id=row["job_id"], project_id=row["project_id"], status=row["status"], stage=row["stage"],
            progress=row["progress"], attempt=row["attempt"], request_data=json.loads(row["request_json"]),
            result_data=json.loads(row["result_json"]), parent_job_id=row["parent_job_id"],
            cancel_requested=bool(row["cancel_requested"]), created_at=row["created_at"], updated_at=row["updated_at"],
            started_at=row["started_at"], finished_at=row["finished_at"], error_type=row["error_type"], error_message=row["error_message"],
        )

    async def create_job(self, *, job_id: str, project_id: int, request_data: dict[str, Any],
                         attempt: int = 1, parent_job_id: str | None = None) -> IngestionJobRecord:
        now = utc_now_iso()
        async with self._lock:
            await self.connection.execute(
                "INSERT INTO ingestion_jobs(job_id,project_id,status,stage,progress,attempt,request_json,result_json,parent_job_id,cancel_requested,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, project_id, IngestionJobStatus.PENDING, IngestionStage.QUEUED, 0.0, attempt,
                 json.dumps(request_data, ensure_ascii=False), "{}", parent_job_id, 0, now, now),
            )
            await self.connection.commit()
        return await self.get_job(job_id)

    async def get_job(self, job_id: str) -> IngestionJobRecord:
        cursor = await self.connection.execute("SELECT * FROM ingestion_jobs WHERE job_id=?", (job_id,))
        row = await cursor.fetchone()
        if row is None:
            raise IngestionJobNotFoundError(f"Ingestion Job 不存在：{job_id}")
        return self._record(row)

    async def update_state(self, *, job_id: str, status: IngestionJobStatus | None = None,
                           stage: IngestionStage | None = None, progress: float | None = None,
                           result_data: dict[str, Any] | None = None, cancel_requested: bool | None = None,
                           error_type: str | None = None, error_message: str | None = None,
                           started_at: str | None = None, finished_at: str | None = None) -> IngestionJobRecord:
        job = await self.get_job(job_id)
        updates: dict[str, Any] = {"updated_at": utc_now_iso()}
        if status is not None: updates["status"] = status
        if stage is not None: updates["stage"] = stage
        if progress is not None:
            if not 0 <= progress <= 1: raise ValueError("progress 必须在 0 到 1 之间")
            updates["progress"] = progress
        if result_data is not None: updates["result_json"] = json.dumps(result_data, ensure_ascii=False)
        if cancel_requested is not None: updates["cancel_requested"] = int(cancel_requested)
        if error_type is not None: updates["error_type"] = error_type
        if error_message is not None: updates["error_message"] = error_message
        if started_at is not None: updates["started_at"] = started_at
        if finished_at is not None: updates["finished_at"] = finished_at
        async with self._lock:
            columns = ", ".join(f"{key}=?" for key in updates)
            await self.connection.execute(f"UPDATE ingestion_jobs SET {columns} WHERE job_id=?", (*updates.values(), job.job_id))
            await self.connection.commit()
        return await self.get_job(job_id)

    async def mark_orphaned_jobs_failed(self) -> int:
        now = utc_now_iso()
        async with self._lock:
            cursor = await self.connection.execute(
                "UPDATE ingestion_jobs SET status=?, finished_at=?, updated_at=?, error_type=?, error_message=? WHERE status IN (?,?)",
                (IngestionJobStatus.FAILED, now, now, "ProcessRestarted", "服务重启，原进程中的后台任务已停止",
                 IngestionJobStatus.RUNNING, IngestionJobStatus.CANCELLING),
            )
            await self.connection.commit()
        return cursor.rowcount
