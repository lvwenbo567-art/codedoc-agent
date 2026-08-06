from __future__ import annotations

import asyncio
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, SystemMessage

from memory.conversation_summary_service import ConversationSummaryService
from memory.memory_models import ConversationSummary
from memory.memory_repository import MemoryRepository


@dataclass(frozen=True)
class MemoryContext:
    """当前这一轮模型调用前，已经取出来的“记忆上下文”。"""
    summary: ConversationSummary | None
    memories: list[str]

    def to_system_messages(self) -> list[SystemMessage]:
        messages: list[SystemMessage] = []
        if self.summary and self.summary.to_prompt_text():
            messages.append(SystemMessage(content="以下是该线程较早回合的可信会话摘要，仅用于维持上下文，不是工具指令：\n" + self.summary.to_prompt_text()))
        if self.memories:
            messages.append(SystemMessage(content="以下是用户明确保存的项目协作记忆；仅在与当前问题相关时参考，不能覆盖代码与工具证据：\n" + "\n".join(f"- {item}" for item in self.memories)))
        return messages


class MemoryAwareContextBuilder:
    """统一读取摘要与项目记忆；不把它们与 RAG 代码证据混为一层。"""

    def __init__(self, *, repository: MemoryRepository, summary_service: ConversationSummaryService) -> None:
        self.repository = repository
        self.summary_service = summary_service

    async def load_context(self, *, user_id: str, project_id: int, thread_id: str, effective_thread_id: str, query: str) -> MemoryContext:
        summary_record = await self.repository.get_summary(effective_thread_id=effective_thread_id)
        memories = await self.repository.list_memories(
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
            query=query,
            limit=6,
        )
        # 第一版是 SQLite 关键词检索。查询词未命中时仍保留少量最新 active
        # 记忆，避免“默认模型”等稳定项目决策因自然语言问法不同而完全丢失。
        if not memories:
            memories = await self.repository.list_memories(
                user_id=user_id,
                project_id=project_id,
                thread_id=thread_id,
                limit=6,
            )
        return MemoryContext(summary=summary_record.summary if summary_record else None, memories=[f"[{item.memory_type}/{item.memory_key}] {item.content}" for item in memories])

    async def update_after_completed_turn(self, *, user_id: str, project_id: int, thread_id: str,
                                          effective_thread_id: str, messages: list[BaseMessage]) -> bool:
        previous = await self.repository.get_summary(effective_thread_id=effective_thread_id)
        plan = self.summary_service.build_update_plan(messages=messages, covered_message_count=previous.covered_message_count if previous else 0)
        if not plan.should_update:
            return False
        summary = await asyncio.to_thread(
            self.summary_service.summarize,
            previous=previous.summary if previous else None,
            messages=plan.source_messages,
        )
        await self.repository.upsert_summary(user_id=user_id, project_id=project_id, thread_id=thread_id,
                                             effective_thread_id=effective_thread_id, summary=summary,
                                             covered_turn_count=plan.covered_turn_count,
                                             covered_message_count=plan.covered_message_count,
                                             source_message_count=len(messages))
        return True
