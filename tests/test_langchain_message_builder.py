from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_agent.message_builder import ConversationTurn, build_chat_messages


def test_build_chat_messages_with_history_limit():
    history = [
        ConversationTurn(role="user", content="第一轮问题"),
        ConversationTurn(role="assistant", content="第一轮回答"),
        ConversationTurn(role="user", content="第二轮问题"),
    ]

    messages = build_chat_messages(
        query="现在的问题",
        history=history,
        max_history_messages=2,
    )

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], HumanMessage)
    assert isinstance(messages[3], HumanMessage)
    assert messages[1].content == "第一轮回答"
    assert messages[2].content == "第二轮问题"
    assert messages[3].content == "现在的问题"


def test_build_chat_messages_without_history():
    messages = build_chat_messages(
        query="解释项目结构",
        history=[],
    )

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)


def test_build_chat_messages_rejects_empty_query():
    try:
        build_chat_messages(query="   ")
    except ValueError as exc:
        assert "query 不能为空" in str(exc)
        return

    raise AssertionError("空 query 应当抛出异常")

