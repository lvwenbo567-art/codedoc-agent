from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from memory.conversation_summary_service import ConversationSummaryService
from memory.memory_context_service import MemoryAwareContextBuilder
from memory.memory_models import ConversationSummary, CreateMemoryInput
from memory.memory_repository import MemoryRepository


@pytest.mark.asyncio
async def test_context_builder_keeps_project_memory_separate_from_summary(tmp_path) -> None:
    repository = MemoryRepository(database_path=str(tmp_path / "memory.sqlite"))
    await repository.start()
    try:
        await repository.upsert_summary(
            user_id="u1", project_id=1, thread_id="chat", effective_thread_id="project:1:thread:chat",
            summary=ConversationSummary(user_goal="理解 RAG"), covered_turn_count=2,
            covered_message_count=4, source_message_count=6,
        )
        await repository.create_memory(CreateMemoryInput(
            user_id="u1", project_id=1, memory_scope="project", memory_type="project_decision",
            memory_key="default_chat_model", content="qwen3.5:4b",
        ))
        builder = MemoryAwareContextBuilder(repository=repository, summary_service=ConversationSummaryService(model=None))
        context = await builder.load_context(user_id="u1", project_id=1, thread_id="chat",
                                             effective_thread_id="project:1:thread:chat", query="使用什么模型")
        system_messages = context.to_system_messages()
        assert len(system_messages) == 2
        assert "理解 RAG" in str(system_messages[0].content)
        assert "qwen3.5:4b" in str(system_messages[1].content)
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_completed_long_history_updates_summary(tmp_path) -> None:
    repository = MemoryRepository(database_path=str(tmp_path / "memory.sqlite"))
    await repository.start()
    try:
        builder = MemoryAwareContextBuilder(repository=repository, summary_service=ConversationSummaryService(
            model=None, trigger_messages=3, trigger_tokens=10000, keep_recent_turns=1,
        ))
        messages = [HumanMessage(content=f"第 {index} 轮问题") for index in range(4)]
        updated = await builder.update_after_completed_turn(user_id="u1", project_id=1, thread_id="chat",
                                                            effective_thread_id="project:1:thread:chat", messages=messages)
        assert updated is True
        record = await repository.get_summary(effective_thread_id="project:1:thread:chat")
        assert record is not None
        assert record.covered_turn_count == 3
        assert record.covered_message_count == 3
    finally:
        await repository.close()
