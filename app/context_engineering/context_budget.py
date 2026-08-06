from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBudgetConfig:#计划最多允许多少”；
    """一次模型调用的上下文预算。"""

    max_context_tokens: int = 16000#模型可接受的总上下文上限。
    reserved_output_tokens: int = 2000#给模型回答预留的空间。若不预留，输入塞满后模型可能无法生成答案。
    max_message_tokens: int = 4000#用户多轮聊天历史的最大预算。
    max_evidence_tokens: int = 7000#RAG 证据的总预算。
    max_single_evidence_tokens: int = 1200#一条证据最长只能占多少，防止单个超长文件垄断上下文。
    max_evidence_items: int = 8#最多使用多少条证据。
    max_items_per_source: int = 2#同一个文件最多贡献几条，增加来源多样性。

    def validate(self) -> None:
        values = {
            "max_context_tokens": self.max_context_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "max_message_tokens": self.max_message_tokens,
            "max_evidence_tokens": self.max_evidence_tokens,
            "max_single_evidence_tokens": self.max_single_evidence_tokens,
            "max_evidence_items": self.max_evidence_items,
            "max_items_per_source": self.max_items_per_source,
        }
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.reserved_output_tokens >= self.max_context_tokens:
            raise ValueError("输出预留 Token 必须小于总 Context Budget")
        if (
            self.max_message_tokens
            + self.max_evidence_tokens
            + self.reserved_output_tokens
            > self.max_context_tokens
        ):
            '''
            小于 16,000，因此合法，还剩约 3,000 Token 给：
            system prompt；
            工具定义 JSON Schema；
            用户本轮问题；
            消息协议本身的额外开销。
            '''
            raise ValueError("消息、证据和输出预留预算之和超过总 Context Budget")


@dataclass(frozen=True)
class ContextBudgetUsage:#“本次实际各部分占了多少”。
    system_tokens: int
    tool_schema_tokens: int
    message_tokens: int
    evidence_tokens: int
    reserved_output_tokens: int

    @property
    def total_input_tokens(self) -> int:
        return (
            self.system_tokens
            + self.tool_schema_tokens
            + self.message_tokens
            + self.evidence_tokens
        )

    @property
    def total_with_output_reserve(self) -> int:
        return self.total_input_tokens + self.reserved_output_tokens
