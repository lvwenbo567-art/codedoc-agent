from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from memory.conversation_summary_service import ConversationSummaryService


def _turn(index: int):
    return [
        HumanMessage(content=f"问题 {index}"),
        AIMessage(content="", tool_calls=[{"name": "search_code", "args": {"query": "x"}, "id": f"call-{index}"}]),
        ToolMessage(content=f"工具结果 {index}", tool_call_id=f"call-{index}"),
        AIMessage(content=f"回答 {index}"),
    ]


def test_summary_plan_keeps_latest_complete_turns_and_never_splits_tool_messages() -> None:
    messages = [message for index in range(5) for message in _turn(index)]
    service = ConversationSummaryService(model=None, trigger_messages=10, trigger_tokens=10000, keep_recent_turns=2)
    plan = service.build_update_plan(messages=messages, covered_message_count=0)
    assert plan.should_update is True
    assert len(plan.source_messages) == 12
    assert len(plan.kept_recent_messages) == 8
    assert isinstance(plan.kept_recent_messages[0], HumanMessage)
    assert isinstance(plan.kept_recent_messages[2], ToolMessage)


def test_summary_skips_tool_content_and_redacts_secret() -> None:
    messages = [
        HumanMessage(content="请记住 Authorization: Bearer secret-value"),
        ToolMessage(content="非常长的工具输出，不应进入摘要", tool_call_id="call-1"),
    ]
    summary = ConversationSummaryService(model=None).summarize(previous=None, messages=messages)
    assert "secret-value" not in summary.user_goal
    assert "[REDACTED]" in summary.user_goal
