from __future__ import annotations

from dataclasses import dataclass

from security.prompt_injection_detector import PromptInjectionDetector
from security.sensitive_data_redactor import SensitiveDataRedactor


@dataclass(frozen=True)
class MemoryPolicyResult:
    allowed: bool
    content: str
    error_message: str | None = None


class MemoryWritePolicy:
    """长期记忆只能保存人工确认的、非敏感且非注入式内容。"""

    def validate_manual_content(self, content: str) -> MemoryPolicyResult:
        normalized = content.strip()
        if not normalized:
            return MemoryPolicyResult(False, "", "记忆内容不能为空")

        injection = PromptInjectionDetector().scan(normalized)
        if injection.suspicious:
            return MemoryPolicyResult(False, "", "疑似 Prompt Injection，拒绝写入长期记忆")

        redaction = SensitiveDataRedactor().redact(normalized)
        if redaction.redacted_count:
            return MemoryPolicyResult(False, "", "内容包含敏感凭据，拒绝写入长期记忆")

        return MemoryPolicyResult(True, normalized)
