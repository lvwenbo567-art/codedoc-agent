from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx

from config import (
    DEFAULT_CHAT_API_KEY,
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_CHAT_MAX_TOKENS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_TIMEOUT_SECONDS,
)


@dataclass(frozen=True)
class ChatConfig:
    """
    Chat model call configuration.
    """

    provider: str = DEFAULT_CHAT_PROVIDER
    model_name: str = DEFAULT_CHAT_MODEL
    base_url: str = DEFAULT_CHAT_BASE_URL
    api_key: str = DEFAULT_CHAT_API_KEY
    timeout_seconds: float = DEFAULT_CHAT_TIMEOUT_SECONDS
    temperature: float = DEFAULT_CHAT_TEMPERATURE
    max_tokens: int = DEFAULT_CHAT_MAX_TOKENS

    def validate(self) -> None:
        """
        Validate chat configuration before sending a request.
        """
        if self.provider not in {"mock", "openai_compatible"}:
            raise ValueError(f"unsupported chat provider: {self.provider}")

        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        if self.provider == "openai_compatible" and not self.base_url.strip():
            raise ValueError("openai_compatible provider requires base_url")


class ChatClient:
    """
    Chat model client.

    Supported providers:
    1. mock
    2. openai_compatible
    """

    def __init__(
        self,
        config: ChatConfig,
        http_client: Optional[httpx.Client] = None,
    ):
        config.validate()

        self.config = config
        self.http_client = http_client

    def generate(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Generate an answer from chat messages.
        """
        self._validate_messages(messages)

        if self.config.provider == "mock":
            return self._generate_mock(messages)

        return self._generate_openai_compatible(messages)

    def _validate_messages(
        self,
        messages: List[Dict[str, str]],
    ) -> None:
        """
        Validate chat messages.
        """
        if not messages:
            raise ValueError("messages cannot be empty")

        valid_roles = {"system", "user", "assistant"}

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role not in valid_roles:
                raise ValueError(f"unsupported message role: {role}")

            if not isinstance(content, str):
                raise ValueError("message content must be a string")

            if not content.strip():
                raise ValueError("message content cannot be empty")

    def _generate_mock(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Generate a deterministic mock answer for local tests.
        """
        user_message = messages[-1]["content"]

        if "[Source 1]" not in user_message:
            return (
                "当前检索结果不足，无法根据项目内容可靠回答该问题。"
            )

        return (
            "根据当前检索到的项目内容，可以从最相关的代码或文档片段中"
            f"分析该问题。当前使用的模型是 {self.config.model_name}。[Source 1]"
        )

    def _generate_openai_compatible(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Call an OpenAI-compatible /chat/completions endpoint.
        """
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }

        owns_client = self.http_client is None
        client = self.http_client or httpx.Client(
            timeout=self.config.timeout_seconds,
            trust_env=False,
        )

        try:
            response = client.post(
                url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        except httpx.TimeoutException as exc:
            raise TimeoutError("chat model request timed out") from exc

        except httpx.RequestError as exc:
            raise ConnectionError(
                f"cannot connect to chat model service: {exc}"
            ) from exc

        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            response_text = exc.response.text.strip()
            if len(response_text) > 500:
                response_text = response_text[:500] + "..."

            detail = f"chat model service returned status code: {status_code}"
            if response_text:
                detail = f"{detail}; response body: {response_text}"

            raise RuntimeError(
                detail
            ) from exc

        finally:
            if owns_client:
                client.close()

        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("chat model response schema is invalid") from exc

        if not isinstance(answer, str):
            raise ValueError("chat model response content is not a string")

        if not answer.strip():
            raise ValueError("chat model returned an empty answer")

        return answer.strip()


def generate_chat_response(
    messages: List[Dict[str, str]],
    provider: str = DEFAULT_CHAT_PROVIDER,
    model_name: str = DEFAULT_CHAT_MODEL,
    base_url: str = DEFAULT_CHAT_BASE_URL,
    api_key: str = DEFAULT_CHAT_API_KEY,
    timeout_seconds: float = DEFAULT_CHAT_TIMEOUT_SECONDS,
    temperature: float = DEFAULT_CHAT_TEMPERATURE,
    max_tokens: int = DEFAULT_CHAT_MAX_TOKENS,
) -> str:
    """
    Unified chat model entrypoint for services.
    """
    config = ChatConfig(
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    client = ChatClient(config=config)

    return client.generate(messages)
