from __future__ import annotations

import pytest

from memory.memory_models import ConversationSummary, CreateMemoryInput, UpdateMemoryInput
from memory.memory_repository import MemoryRepository


@pytest.mark.asyncio
async def test_summary_is_isolated_by_effective_thread_id(tmp_path) -> None:
    repository = MemoryRepository(database_path=str(tmp_path / "memory.sqlite"))
    await repository.start()
    try:
        await repository.upsert_summary(
            user_id="u1", project_id=1, thread_id="a", effective_thread_id="project:1:thread:a",
            summary=ConversationSummary(user_goal="理解检索"), covered_turn_count=2,
            covered_message_count=4, source_message_count=6,
        )
        assert await repository.get_summary(effective_thread_id="project:1:thread:b") is None
        record = await repository.get_summary(effective_thread_id="project:1:thread:a")
        assert record is not None
        assert record.summary.user_goal == "理解检索"
        assert record.covered_message_count == 4
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_project_memory_supersedes_old_active_key(tmp_path) -> None:
    repository = MemoryRepository(database_path=str(tmp_path / "memory.sqlite"))
    await repository.start()
    try:
        first = await repository.create_memory(CreateMemoryInput(
            user_id="u1", project_id=1, memory_scope="project", memory_type="project_decision",
            memory_key="default_chat_model", content="qwen3.5:4b",
        ))
        second = await repository.create_memory(CreateMemoryInput(
            user_id="u1", project_id=1, memory_scope="project", memory_type="project_decision",
            memory_key="default_chat_model", content="qwen3.5:9b",
        ))
        active = await repository.list_memories(user_id="u1", project_id=1)
        assert [item.memory_id for item in active] == [second.memory_id]
        old = await repository.get_memory(memory_id=first.memory_id, user_id="u1", project_id=1)
        assert old.status == "superseded"
        assert old.superseded_by == second.memory_id
        updated = await repository.update_memory(memory_id=second.memory_id, user_id="u1", project_id=1,
                                                 value=UpdateMemoryInput(content="qwen3.5:4b"))
        assert updated.version == 2
    finally:
        await repository.close()
