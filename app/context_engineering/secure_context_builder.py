from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from context_engineering.context_budget import ContextBudgetConfig
from context_engineering.evidence_selector import select_evidence
from context_engineering.token_counter import ApproximateTokenCounter, TokenCounter
from security.prompt_injection_detector import PromptInjectionDetector
from security.sensitive_data_redactor import SensitiveDataRedactor


@dataclass(frozen=True)
class SecureContextResult:
    context: str#最终要放进 Prompt 的文本
    selected_evidence: list[dict[str, Any]]#实际保留的结构化 Chunk
    evidence_tokens: int
    injection_warning_count: int#发现多少条可疑注入证据
    redacted_count: int#一共替换掉多少处敏感信息
    dropped_evidence_count: int#因去重、超预算等丢掉多少条证据


class SecureContextBuilder:
    """将检索结果转换为有预算、脱敏且标记为不可信的模型上下文。"""

    def __init__(
        self,
        *,
        budget: ContextBudgetConfig | None = None,
        token_counter: TokenCounter | None = None,
        injection_detector: PromptInjectionDetector | None = None,
        redactor: SensitiveDataRedactor | None = None,
    ) -> None:
        self.budget = budget or ContextBudgetConfig()
        self.budget.validate()
        self.token_counter = token_counter or ApproximateTokenCounter()
        self.injection_detector = injection_detector or PromptInjectionDetector()
        self.redactor = redactor or SensitiveDataRedactor()

    def build(self, *, evidence_items: list[dict[str, Any]]) -> SecureContextResult:
        '''复制并清洗每一条证据'''
        sanitized_items: list[dict[str, Any]] = []#清洗后的 Chunk 列表
        injection_warning_count = 0#可疑 Chunk 的数量
        redacted_count = 0#被脱敏命中的敏感内容总数量
        for evidence in evidence_items:
            item = dict(evidence)
            content = str(item.get("content") or "")
            scan = self.injection_detector.scan(content)
            redaction = self.redactor.redact(content)
            item["content"] = redaction.text
            item["prompt_injection_suspicious"] = scan.suspicious
            item["prompt_injection_risk_score"] = scan.risk_score
            item["prompt_injection_findings"] = [finding.rule_name for finding in scan.findings]
            injection_warning_count += int(scan.suspicious)
            redacted_count += redaction.redacted_count
            sanitized_items.append(item)

        selection = select_evidence(
            evidence_items=sanitized_items,
            token_counter=self.token_counter,
            max_total_tokens=self.budget.max_evidence_tokens,
            max_single_tokens=self.budget.max_single_evidence_tokens,
            max_items=self.budget.max_evidence_items,
            max_items_per_source=self.budget.max_items_per_source,
        )

        blocks: list[str] = []#blocks 是每条证据的 Prompt 文本块。
        for item in selection.selected:
            metadata = {
                "chunk_id": item.get("chunk_id"),
                "source_path": item.get("source_path"),
                "chunk_type": item.get("chunk_type"),
                "trust_level": "untrusted",
                "injection_suspicious": item.get("prompt_injection_suspicious", False),
            }
            blocks.append(
                "[UNTRUSTED_EVIDENCE]\n"
                + json.dumps(metadata, ensure_ascii=False)
                + "\n"
                + str(item.get("content") or "")
                + "\n[END_UNTRUSTED_EVIDENCE]"
            )

        return SecureContextResult(
            context="\n\n".join(blocks),
            selected_evidence=selection.selected,
            evidence_tokens=selection.selected_tokens,
            injection_warning_count=injection_warning_count,
            redacted_count=redacted_count,
            dropped_evidence_count=selection.dropped_count,
        )
