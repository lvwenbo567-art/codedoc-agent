from security.sensitive_data_redactor import SensitiveDataRedactor
from langchain_core.messages import ToolMessage
from langgraph_agent.tool_agent_nodes import redact_tool_messages


def test_redactor_removes_common_credentials() -> None:
    result = SensitiveDataRedactor().redact(
        "Authorization: Bearer abc123 API_KEY=secret PASSWORD: pass"
    )
    assert "abc123" not in result.text
    assert "secret" not in result.text
    assert result.redacted_count == 3


def test_tool_message_is_redacted_before_next_model_call() -> None:
    messages = redact_tool_messages(
        [ToolMessage(content="Authorization: Bearer abc123", tool_call_id="call_1")]
    )
    assert "abc123" not in str(messages[0].content)
