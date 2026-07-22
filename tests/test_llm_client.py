from pathlib import Path
import sys
import json

import httpx
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from clients.llm_client import ChatClient, ChatConfig, generate_chat_response


def build_messages():
    return [
        {
            "role": "system",
            "content": "You answer from retrieved context.",
        },
        {
            "role": "user",
            "content": "Question\n\n[Source 1]\ncontent",
        },
    ]


def test_mock_chat_client_returns_cited_answer():
    answer = generate_chat_response(
        messages=build_messages(),
        provider="mock",
        model_name="test-chat",
    )

    assert "test-chat" in answer
    assert "[Source 1]" in answer


def test_mock_chat_client_handles_no_source_context():
    answer = generate_chat_response(
        messages=[
            {
                "role": "user",
                "content": "no retrieved context",
            }
        ],
        provider="mock",
    )

    assert "[Source 1]" not in answer
    assert "不足" in answer


def test_chat_config_rejects_invalid_provider():
    with pytest.raises(ValueError):
        ChatConfig(provider="unknown").validate()


def test_chat_client_rejects_invalid_messages():
    client = ChatClient(
        config=ChatConfig(provider="mock"),
    )

    with pytest.raises(ValueError):
        client.generate([])

    with pytest.raises(ValueError):
        client.generate(
            [
                {
                    "role": "tool",
                    "content": "hello",
                }
            ]
        )

    with pytest.raises(ValueError):
        client.generate(
            [
                {
                    "role": "user",
                    "content": "   ",
                }
            ]
        )


def test_openai_compatible_sends_expected_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["payload"] = request.read().decode("utf-8")

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "OpenAI-compatible answer [Source 1]",
                        }
                    }
                ]
            },
        )

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    client = ChatClient(
        config=ChatConfig(
            provider="openai_compatible",
            model_name="qwen-test",
            base_url="http://llm.local/v1/",
            api_key="test-key",
            temperature=0.3,
            max_tokens=123,
        ),
        http_client=http_client,
    )

    answer = client.generate(build_messages())

    assert answer == "OpenAI-compatible answer [Source 1]"
    assert captured["url"] == "http://llm.local/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    payload = json.loads(captured["payload"])
    assert payload["model"] == "qwen-test"
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 123
    assert payload["stream"] is False

    http_client.close()


def test_openai_compatible_maps_timeout_to_timeout_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    client = ChatClient(
        config=ChatConfig(
            provider="openai_compatible",
            base_url="http://llm.local/v1",
        ),
        http_client=http_client,
    )

    with pytest.raises(TimeoutError):
        client.generate(build_messages())

    http_client.close()


def test_openai_compatible_rejects_invalid_response_schema():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    client = ChatClient(
        config=ChatConfig(
            provider="openai_compatible",
            base_url="http://llm.local/v1",
        ),
        http_client=http_client,
    )

    with pytest.raises(ValueError):
        client.generate(build_messages())

    http_client.close()


def test_openai_compatible_includes_error_response_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            text='{"error":"model runner failed"}',
            request=request,
        )

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
    )

    client = ChatClient(
        config=ChatConfig(
            provider="openai_compatible",
            base_url="http://llm.local/v1",
        ),
        http_client=http_client,
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.generate(build_messages())

    assert "状态码：502" in str(exc_info.value)
    assert "model runner failed" in str(exc_info.value)

    http_client.close()
