from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from langchain_agent.trace_recorder import AgentTraceRecorder


def _tool_call_ids(message: BaseMessage) -> set[str]:
    if not isinstance(message, AIMessage):
        return set()

    ids: set[str] = set()

    for tool_call in message.tool_calls or []:
        call_id = str(tool_call.get("id") or "")

        if call_id:
            ids.add(call_id)

    return ids


def select_message_window(
    *,
    messages: list[BaseMessage],
    max_messages: int,
) -> list[BaseMessage]:
    """
    按消息窗口裁剪历史，同时避免破坏 AI tool_calls 与 ToolMessage 的配对。
    """
    if max_messages <= 0:
        raise ValueError("max_messages 必须大于 0")

    if len(messages) <= max_messages:
        return list(messages)

    selected = list(messages[-max_messages:])
    known_tool_call_ids: set[str] = set()

    for message in selected:
        known_tool_call_ids.update(
            _tool_call_ids(message)
        )

    index = 0

    while index < len(selected):
        message = selected[index]

        if not isinstance(message, ToolMessage):
            index += 1
            continue

        tool_call_id = str(message.tool_call_id or "")

        if tool_call_id in known_tool_call_ids:
            index += 1
            continue

        original_index = messages.index(message)
        paired_ai_message = None

        for previous in reversed(messages[:original_index]):
            if tool_call_id in _tool_call_ids(previous):
                paired_ai_message = previous
                break

        if paired_ai_message is None:
            selected.pop(index)
            continue

        selected.insert(index, paired_ai_message)
        known_tool_call_ids.update(
            _tool_call_ids(paired_ai_message)
        )
        index += 2

    if len(selected) > max_messages:
        overflow = len(selected) - max_messages
        selected = selected[overflow:]

        while (
            selected
            and isinstance(selected[0], ToolMessage)
        ):
            selected = selected[1:]

    return selected


class CodeDocMessageWindowMiddleware:
    """
    项目层消息窗口裁剪器。

    当前版本不强依赖 LangChain 内部 middleware API，
    由 AgentService 在调用模型前显式使用。
    """

    def __init__(
        self,
        *,
        max_messages: int,
        recorder: AgentTraceRecorder | None = None,
    ) -> None:
        self.max_messages = max_messages
        self.recorder = recorder

    def trim(
        self,
        messages: list[BaseMessage],
    ) -> list[BaseMessage]:
        selected = select_message_window(
            messages=messages,
            max_messages=self.max_messages,
        )

        if (
            self.recorder is not None
            and len(selected) != len(messages)
        ):
            self.recorder.add_message_trim(
                original_count=len(messages),
                kept_count=len(selected),
            )

        return selected
