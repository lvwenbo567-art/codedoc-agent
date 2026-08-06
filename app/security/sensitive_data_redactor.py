from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RedactionResult:
    text: str
    redacted_count: int


class SensitiveDataRedactor:
    """在内容进入模型、Trace 或 SSE 前移除常见凭据。"""

    _patterns = (
        
        re.compile(r"(Authorization\s*:\s*Bearer\s+)[^\s\"']+", re.I),
        re.compile(r"\b(API[_-]?KEY|TOKEN|PASSWORD|SECRET)\s*[:=]\s*[^\s\"']+", re.I),
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", re.I),
    )

    def redact(self, text: str) -> RedactionResult:
        redacted_count = 0
        result = text
        for pattern in self._patterns:
            def replace(match: re.Match[str]) -> str:
                nonlocal redacted_count
                redacted_count += 1
                prefix = match.group(1) if match.lastindex else ""
                return f"{prefix}[REDACTED]"
            result = pattern.sub(replace, result)
        return RedactionResult(text=result, redacted_count=redacted_count)
