from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import aiosqlite

from memory.memory_models import (
    ConversationSummary,
    ConversationSummaryRecord,
    CreateMemoryInput,
    MemoryItem,
    UpdateMemoryInput,
    utc_now_iso,
)


class MemoryNotFoundError(ValueError):
    pass


class MemoryRepository:
    """应用自有 SQLite 表；不触碰 LangGraph Checkpointer 的内部表。"""

    def __init__(self, *, database_path: str) -> None:
        self.database_path = str(Path(database_path).resolve())
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("MemoryRepository 尚未启动")
        return self._connection

    async def start(self) -> None:
        if self._connection is not None:
            return
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.database_path)
        self._connection.row_factory = aiosqlite.Row
        await self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                user_id TEXT NOT NULL,
                project_id INTEGER NOT NULL,
                thread_id TEXT NOT NULL,
                effective_thread_id TEXT NOT NULL PRIMARY KEY,
                summary_json TEXT NOT NULL,
                covered_turn_count INTEGER NOT NULL DEFAULT 0,
                covered_message_count INTEGER NOT NULL DEFAULT 0,
                source_message_count INTEGER NOT NULL DEFAULT 0,
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conversation_summaries_scope
              ON conversation_summaries(user_id, project_id, thread_id);

            CREATE TABLE IF NOT EXISTS memory_items (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                project_id INTEGER NOT NULL,
                thread_id TEXT,
                memory_scope TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_reference TEXT,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                version INTEGER NOT NULL,
                superseded_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT,
                last_accessed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memory_items_scope
              ON memory_items(user_id, project_id, memory_scope, status);
            CREATE INDEX IF NOT EXISTS idx_memory_items_key
              ON memory_items(user_id, project_id, memory_key, status);
            """
        )
        await self.connection.commit()

    async def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()

    @staticmethod
    def _summary_record(row: aiosqlite.Row) -> ConversationSummaryRecord:
        return ConversationSummaryRecord(
            user_id=row["user_id"], project_id=row["project_id"], thread_id=row["thread_id"],
            effective_thread_id=row["effective_thread_id"], summary=ConversationSummary.model_validate(json.loads(row["summary_json"])),
            covered_turn_count=row["covered_turn_count"], covered_message_count=row["covered_message_count"],
            source_message_count=row["source_message_count"], version=row["version"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _memory_item(row: aiosqlite.Row) -> MemoryItem:
        return MemoryItem(**dict(row))#把数据库行转为字典,把字典展开成关键字参数：

    async def get_summary(self, *, effective_thread_id: str) -> ConversationSummaryRecord | None:
        cursor = await self.connection.execute("SELECT * FROM conversation_summaries WHERE effective_thread_id=?", (effective_thread_id,))
        row = await cursor.fetchone()
        return self._summary_record(row) if row else None

    async def upsert_summary(self, *, user_id: str, project_id: int, thread_id: str, effective_thread_id: str,
                             summary: ConversationSummary, covered_turn_count: int,
                             covered_message_count: int, source_message_count: int) -> ConversationSummaryRecord:
        existing = await self.get_summary(effective_thread_id=effective_thread_id)
        now = utc_now_iso()
        version = (existing.version + 1) if existing else 1
        created_at = existing.created_at if existing else now
        async with self._lock:
            await self.connection.execute(
                """INSERT INTO conversation_summaries(user_id,project_id,thread_id,effective_thread_id,summary_json,
                    covered_turn_count,covered_message_count,source_message_count,version,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(effective_thread_id) DO UPDATE SET summary_json=excluded.summary_json,
                     covered_turn_count=excluded.covered_turn_count, covered_message_count=excluded.covered_message_count,
                     source_message_count=excluded.source_message_count, version=excluded.version, updated_at=excluded.updated_at""",
                (user_id, project_id, thread_id, effective_thread_id, summary.model_dump_json(), covered_turn_count,
                 covered_message_count, source_message_count, version, created_at, now),
            )
            await self.connection.commit()
        result = await self.get_summary(effective_thread_id=effective_thread_id)
        assert result is not None
        return result

    async def create_memory(self, value: CreateMemoryInput) -> MemoryItem:
        now = utc_now_iso()
        item = MemoryItem(memory_id=str(uuid.uuid4()), **value.model_dump(), created_at=now, updated_at=now)
        async with self._lock:
            if item.memory_scope in {"user", "project"}:
                await self.connection.execute(
                    "UPDATE memory_items SET status='superseded', superseded_by=?, updated_at=? WHERE user_id=? AND project_id=? AND memory_scope=? AND memory_key=? AND status='active'",
                    (item.memory_id, now, item.user_id, item.project_id, item.memory_scope, item.memory_key),
                )
            await self.connection.execute(
                """INSERT INTO memory_items VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                tuple(item.model_dump().values()),
            )
            await self.connection.commit()
        return item

    async def list_memories(self, *, user_id: str, project_id: int, thread_id: str | None = None,
                            query: str | None = None, memory_type: str | None = None,
                            include_inactive: bool = False, limit: int = 20) -> list[MemoryItem]:
        clauses = ["user_id=?", "project_id=?"]
        values: list[object] = [user_id, project_id]
        if not include_inactive:
            clauses.append("status='active'")
        if thread_id:
            clauses.append("(memory_scope!='thread' OR thread_id=?)")
            values.append(thread_id)
        else:
            clauses.append("memory_scope!='thread'")
        if memory_type:
            clauses.append("memory_type=?")
            values.append(memory_type)
        if query:
            clauses.append("(memory_key LIKE ? OR content LIKE ?)")
            values.extend([f"%{query.strip()}%", f"%{query.strip()}%"])
        values.append(limit)
        cursor = await self.connection.execute(
            f"SELECT * FROM memory_items WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT ?", values
        )
        rows = await cursor.fetchall()
        now = utc_now_iso()
        if rows:
            await self.connection.executemany("UPDATE memory_items SET last_accessed_at=? WHERE memory_id=?", [(now, row["memory_id"]) for row in rows])
            await self.connection.commit()
        return [self._memory_item(row) for row in rows]

    async def get_memory(self, *, memory_id: str, user_id: str, project_id: int) -> MemoryItem:
        cursor = await self.connection.execute("SELECT * FROM memory_items WHERE memory_id=? AND user_id=? AND project_id=?", (memory_id, user_id, project_id))
        row = await cursor.fetchone()
        if row is None:
            raise MemoryNotFoundError("记忆不存在或不属于当前用户/项目")
        return self._memory_item(row)

    async def update_memory(self, *, memory_id: str, user_id: str, project_id: int, value: UpdateMemoryInput) -> MemoryItem:
        item = await self.get_memory(memory_id=memory_id, user_id=user_id, project_id=project_id)
        updates = value.model_dump(exclude_unset=True)
        if not updates:
            return item
        updates["updated_at"] = utc_now_iso()
        updates["version"] = item.version + 1
        async with self._lock:
            columns = ", ".join(f"{key}=?" for key in updates)
            await self.connection.execute(f"UPDATE memory_items SET {columns} WHERE memory_id=?", (*updates.values(), memory_id))
            await self.connection.commit()
        return await self.get_memory(memory_id=memory_id, user_id=user_id, project_id=project_id)

    async def delete_memory(self, *, memory_id: str, user_id: str, project_id: int) -> MemoryItem:
        return await self.update_memory(memory_id=memory_id, user_id=user_id, project_id=project_id, value=UpdateMemoryInput(status="deleted"))
