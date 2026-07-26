from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_agent.message_window_middleware import select_message_window


def test_message_window_keeps_tool_pair():
    messages = [
        HumanMessage(content="question"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_code",
                    "args": {
                        "query": "test",
                    },
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="result",
            tool_call_id="call-1",
            name="search_code",
        ),
        AIMessage(content="answer one"),
        HumanMessage(content="follow up"),
        AIMessage(content="answer two"),
    ]

    selected = select_message_window(
        messages=messages,
        max_messages=4,
    )

    for index, message in enumerate(selected):
        if not isinstance(message, ToolMessage):
            continue

        assert index > 0
        assert isinstance(selected[index - 1], AIMessage)
