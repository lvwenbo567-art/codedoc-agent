from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from context_engineering.message_selector import group_messages_by_turn, select_messages_by_budget
from context_engineering.token_counter import CharacterTokenCounter


def test_selector_keeps_tool_protocol_in_one_turn() -> None:
    messages = [
        HumanMessage(content="old"),
        AIMessage(content="old answer"),
        HumanMessage(content="new"),
        AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "call_1"}]),
        ToolMessage(content="result", tool_call_id="call_1"),
        AIMessage(content="final"),
    ]
    selected = select_messages_by_budget(
        messages=messages,
        max_tokens=20,
        token_counter=CharacterTokenCounter(),
    )
    assert selected.messages[-2].__class__ is ToolMessage
    assert selected.dropped_message_count == 2
    assert len(group_messages_by_turn(messages)) == 2
