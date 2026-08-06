from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class InjectionFinding:
    rule_name: str#表示命中了哪条规则。
    matched_text: str#表示正则实际匹配到的原文。
    severity: str#风险等级


@dataclass(frozen=True)
class InjectionScanResult:
    suspicious: bool
    risk_score: int
    findings: list[InjectionFinding]


'''
InjectionScanResult(
    suspicious=True,
    risk_score=8,
    findings=[
        InjectionFinding(
            rule_name="ignore_previous",
            matched_text="Ignore all previous instructions",
            severity="high",
        ),
        InjectionFinding(
            rule_name="reveal_system_prompt",
            matched_text="Reveal the system prompt",
            severity="high",
        ),
    ],
)
'''

INJECTION_RULES: list[tuple[str, re.Pattern[str], str, int]] = [
    ("ignore_previous", re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)", re.I), "high", 4),
    (
        "ignore_previous_chinese",
        re.compile(
            r"(?:忽略|无视|跳过).{0,20}(?:系统|开发者)?(?:指令|提示词|规则)"
        ),
        "high",
        4,
    ),
    ("reveal_system_prompt", re.compile(r"(reveal|show|print|output).{0,30}(system prompt|developer message)", re.I), "high", 4),
    (
        "reveal_system_prompt_chinese",
        re.compile(
            r"(?:泄露|透露|展示|显示|输出|打印).{0,30}"
            r"(?:系统提示词|系统指令|开发者消息)"
        ),
        "high",
        4,
    ),
    ("override_role", re.compile(r"(you are now|act as|new role).{0,50}(admin|system|developer)", re.I), "medium", 2),
    (
        "override_role_chinese",
        re.compile(r"(?:你现在是|扮演|切换为).{0,50}(?:管理员|系统|开发者)"),
        "medium",
        2,
    ),
    ("secret_exfiltration", re.compile(r"(api[_ -]?key|password|token|secret).{0,30}(send|output|reveal|print)", re.I), "high", 4),
    (
        "secret_exfiltration_chinese",
        re.compile(
            r"(?:api[_ -]?key|密码|令牌|token|密钥|secret).{0,30}"
            r"(?:泄露|发送|输出|显示|打印|透露)",
            re.I,
        ),
        "high",
        4,
    ),
    ("tool_override", re.compile(r"(call|invoke|execute).{0,20}(tool|function|shell|command).{0,30}(without|ignore|bypass)", re.I), "high", 4),
    (
        "tool_override_chinese",
        re.compile(
            r"(?:调用|执行|运行).{0,20}(?:工具|函数|命令|shell).{0,30}"
            r"(?:无需|跳过|绕过|忽略)(?:确认|审批|限制)?",
            re.I,
        ),
        "high",
        4,
    ),
]


class PromptInjectionDetector:
    """轻量规则检测器：用于风险标记，不能被视为唯一防线。"""

    def scan(self, text: str) -> InjectionScanResult:
        findings: list[InjectionFinding] = []
        risk_score = 0
        for name, pattern, severity, score in INJECTION_RULES:
            match = pattern.search(text)
            if match:
                findings.append(InjectionFinding(name, match.group(0), severity))
                risk_score += score
        return InjectionScanResult(bool(findings), risk_score, findings)
