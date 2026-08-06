from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_agent.chat_service import extract_message_text
from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.model_factory import create_chat_model
from langgraph_agent.evidence_adapter import evidence_to_retrieved_chunks
from langgraph_agent.state import GraphEvidence
from services.answer_quality import evaluate_answer_quality
from services.citation_builder import build_citations
from services.prompt_builder import build_rag_messages
from context_engineering.secure_context_builder import SecureContextBuilder


@dataclass(frozen=True)
class GraphAnswerResult:
    answer: str
    citations: list[dict]
    answer_quality: dict
    chunks: list[dict]


class GraphAnswerService:
    """
    将 LangGraph evidence 接入已有 RAG 回答链路。

    这里不重写 prompt、citation 和 answer_quality：
    - prompt_builder 负责构造带 Source 的上下文；
    - citation_builder 负责生成 Source 元数据；
    - answer_quality 负责校验回答中的引用。
    """

    def __init__(
        self,
        *,
        model_config: LangChainModelConfig,
        max_context_chars: int | None = None,
    ) -> None:
        self.model_config = model_config
        self.max_context_chars = max_context_chars
        self._model = None

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = create_chat_model(
                self.model_config
            )

        return self._model

    @staticmethod
    def _to_langchain_messages(
        raw_messages: list[dict[str, str]],
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = []

        for message in raw_messages:
            role = message.get("role")
            content = str(
                message.get("content")
                or ""
            )

            if role == "system":
                messages.append(
                    SystemMessage(content=content)
                )

            elif role == "user":
                messages.append(
                    HumanMessage(content=content)
                )

        return messages

    def generate(
        self,
        *,
        query: str,
        evidence: list[GraphEvidence],
        max_context_chars: int,
    ) -> GraphAnswerResult:
        retrieved_chunks = evidence_to_retrieved_chunks(
            evidence
        )

        if not retrieved_chunks:
            raise ValueError(
                "没有可用于生成回答的证据"
            )

        secure_context = SecureContextBuilder().build(
            evidence_items=retrieved_chunks
        )
        retrieved_chunks = secure_context.selected_evidence
        if not retrieved_chunks:
            raise ValueError("安全上下文过滤后没有可用证据")

        raw_messages = build_rag_messages(
            query=query,
            retrieved_chunks=retrieved_chunks,
            max_context_chars=max_context_chars,
        )

        citations = build_citations(
            retrieved_chunks
        )

        if self.model_config.provider == "mock":
            answer = (
                "LangGraph Mock 模式已根据项目证据生成回答。"
                "请在真实模型环境下验证完整自然语言答案 [Source 1]"
            )
        else:
            messages = self._to_langchain_messages(
                raw_messages
            )
            response = self._get_model().invoke(
                messages
            )
            answer = extract_message_text(
                response
            ).strip()

        if not answer:
            raise ValueError(
                "模型返回空回答"
            )

        answer_quality = evaluate_answer_quality(
            answer=answer,
            citations=citations,
        )

        return GraphAnswerResult(
            answer=answer,
            citations=citations,
            answer_quality=answer_quality,
            chunks=retrieved_chunks,
        )

    def build_answer(
        self,
        *,
        query: str,
        evidence: list[GraphEvidence],
    ) -> dict[str, Any]:
        if self.max_context_chars is None:
            raise ValueError(
                "缺少 max_context_chars，无法生成回答"
            )

        result = self.generate(
            query=query,
            evidence=evidence,
            max_context_chars=self.max_context_chars,
        )

        return {
            "answer": result.answer,
            "citations": result.citations,
            "answer_quality": result.answer_quality,
            "chunks": result.chunks,
        }
