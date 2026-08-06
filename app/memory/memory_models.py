from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


MemoryScope = Literal["user", "project", "thread"]#它规定一条长期记忆的作用范围只能是三种之一：
'''
user：用户级记忆
例如：“用户希望回答使用中文。”

project：项目级记忆
例如：“当前项目默认 Chat 模型使用 qwen3.5:4b。”

thread：会话级记忆
例如：“这个线程当前准备继续完善 Memory 系统。”
'''
MemoryType = Literal[
    "user_preference",
    "project_decision",
    "confirmed_fact",
    "open_task",#明确尚未完成的任务。
    "user_correction",
]#它规定“这条记忆是什么性质”。
MemoryStatus = Literal["active", "superseded", "expired", "deleted"]
#最重要的是 superseded，即“被新记忆替代”。

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictMemoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationSummary(StrictMemoryModel):
    """可直接注入模型、也可稳定存入 SQLite 的线程摘要。"""

    user_goal: str = ""
    confirmed_facts: list[str] = Field(default_factory=list)
    project_decisions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    recent_progress: list[str] = Field(default_factory=list)

    def to_prompt_text(self) -> str:
        sections = [
            ("用户目标", [self.user_goal] if self.user_goal else []),
            ("已确认事实", self.confirmed_facts),
            ("项目决策", self.project_decisions),
            ("待解决问题", self.open_questions),
            ("近期进展", self.recent_progress),
        ]
        lines: list[str] = []
        for title, values in sections:
            cleaned = [value.strip() for value in values if value.strip()]
            if cleaned:
                lines.append(f"{title}：")
                lines.extend(f"- {value}" for value in cleaned)
        return "\n".join(lines)


class ConversationSummaryRecord(StrictMemoryModel):
    """这是“数据库中真实保存的一条摘要记录”。"""
    user_id: str
    project_id: int
    thread_id: str
    effective_thread_id: str
    summary: ConversationSummary
    covered_turn_count: int = Field(ge=0)#摘要已经覆盖了多少个完整对话回合；
    covered_message_count: int = Field(ge=0)#已经被摘要覆盖了多少条 Message；
    source_message_count: int = Field(ge=0)#这次生成摘要时，线程总共有多少条 Message。
    version: int = Field(ge=1)
    created_at: str
    updated_at: str


class MemoryItem(StrictMemoryModel):
    """这是长期协作记忆的一条完整记录。数据库最终完整保存的记录"""
    memory_id: str
    user_id: str
    project_id: int
    thread_id: str | None = None
    memory_scope: MemoryScope
    memory_type: MemoryType
    memory_key: str = Field(min_length=1, max_length=120)#类似配置项名称，用来判断两条记忆是不是在描述同一个主题。
    content: str = Field(min_length=1, max_length=4000)#小而明确的事实、偏好、决策
    source_type: str = Field(default="manual", max_length=80)#这条记忆从哪里来。
    source_reference: str | None = Field(default=None, max_length=500)#记录来源的引用信息。
    confidence: float = Field(default=1.0, ge=0, le=1)#记忆可信度。
    status: MemoryStatus = "active"
    version: int = Field(default=1, ge=1)
    superseded_by: str | None = None
    created_at: str
    updated_at: str
    expires_at: str | None = None
    last_accessed_at: str | None = None


class CreateMemoryInput(StrictMemoryModel):
    """用户可以提交的输入"""
    user_id: str = Field(min_length=1, max_length=120)
    project_id: int = Field(ge=1)
    thread_id: str | None = Field(default=None, min_length=1, max_length=120)
    memory_scope: MemoryScope
    memory_type: MemoryType
    memory_key: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=4000)
    source_type: str = Field(default="manual", max_length=80)
    source_reference: str | None = Field(default=None, max_length=500)
    confidence: float = Field(default=1.0, ge=0, le=1)
    expires_at: str | None = None


class UpdateMemoryInput(StrictMemoryModel):
    content: str | None = Field(default=None, min_length=1, max_length=4000)
    memory_type: MemoryType | None = None
    memory_key: str | None = Field(default=None, min_length=1, max_length=120)
    confidence: float | None = Field(default=None, ge=0, le=1)
    expires_at: str | None = None
    status: MemoryStatus | None = None
