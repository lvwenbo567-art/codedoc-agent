from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage

from context_engineering.token_counter import TokenCounter


@dataclass(frozen=True)
class SelectedMessages:
    messages: list[BaseMessage]# 最终保留的消息
    token_count: int#它们约占多少 Token
    dropped_message_count: int#因预算被丢弃了多少条消息


def message_to_text(message: BaseMessage) -> str:
    """把任意 LangChain Message 转换为可计数的文本。"""
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def group_messages_by_turn(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """以 HumanMessage 为新一轮起点，避免拆开 Tool Call 与 ToolMessage。"""
    groups: list[list[BaseMessage]] = []
    current_group: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage) and current_group:
            groups.append(current_group)
            current_group = []
        current_group.append(message)
    if current_group:
        groups.append(current_group)
    return groups


def select_messages_by_budget(
    *,
    messages: list[BaseMessage],
    max_tokens: int,
    token_counter: TokenCounter,
) -> SelectedMessages:
    """按完整对话轮次，从最新消息开始选择不超过预算的历史。"""
    if max_tokens <= 0:
        raise ValueError("max_tokens 必须大于 0")

    selected_groups: list[list[BaseMessage]] = []
    used_tokens = 0
    for group in reversed(group_messages_by_turn(messages)):
        group_tokens = sum(token_counter.count_text(message_to_text(item)) for item in group)
        if selected_groups and used_tokens + group_tokens > max_tokens:
            break
        if not selected_groups and group_tokens > max_tokens:
            selected_groups.append(group)
            used_tokens = group_tokens
            break
        selected_groups.append(group)
        used_tokens += group_tokens

    selected_groups.reverse()
    selected = [message for group in selected_groups for message in group]
    return SelectedMessages(
        messages=selected,
        token_count=used_tokens,
        dropped_message_count=len(messages) - len(selected),
    )
