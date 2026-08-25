from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRequestSecurityDecision:
    """用户请求在进入 ToolNode 前的最小安全与范围判定结果。"""

    allowed: bool
    stop_reason: str | None = None
    error_message: str | None = None


class AgentRequestSecurityPolicy:
    """拦截明显不应由代码仓库 Agent 执行的请求。

    这不是替代 Tool Schema、路径校验或人工审核；它负责在模型调用之后、
    ToolNode 执行之前，对明确的越权、敏感信息和项目无关请求给出确定性边界。
    """

    _destructive_pattern = re.compile(
        r"(?:删除|删掉|移除|清空|格式化|rm\s+-rf|del\s+/[fq]|remove[-_ ]project)",
        re.IGNORECASE,
    )
    _source_mutation_pattern = re.compile(
        r"(?:修改|改为|写入|保存|替换|新增|modify|change|write|save|replace|create)",
        re.IGNORECASE,
    )
    _source_target_pattern = re.compile(
        r"(?:\.py\b|\.json\b|\.ya?ml\b|\.toml\b|\.env\b|"
        r"配置|文件|config|source[_ -]?code)",
        re.IGNORECASE,
    )
    _secret_pattern = re.compile(
        r"(?:读取|查看|显示|导出|泄露|告诉我).{0,30}"
        r"(?:api[_ -]?key|secret|token|password|密码|密钥|访问令牌|\.env)",
        re.IGNORECASE,
    )
    _prompt_injection_pattern = re.compile(
        r"(?:忽略.{0,20}(?:指令|规则|系统)|"
        r"(?:显示|泄露|输出).{0,20}(?:系统提示词|system prompt)|"
        r"(?:读取|查看).{0,20}(?:隐藏文件|系统文件))",
        re.IGNORECASE,
    )
    _unregistered_execution_pattern = re.compile(
        r"(?:部署到生产|发布到生产|deploy[_ -]?production|执行任意(?:命令|shell))",
        re.IGNORECASE,
    )
    _out_of_scope_pattern = re.compile(
        r"(?:今天晚上吃什么|晚饭吃什么|天气怎么样|推荐(?:餐厅|电影)|讲个笑话)",
        re.IGNORECASE,
    )

    def evaluate(self, query: str) -> AgentRequestSecurityDecision:
        normalized = str(query or "").strip()

        if self._prompt_injection_pattern.search(normalized):
            return AgentRequestSecurityDecision(
                allowed=False,
                stop_reason="unsafe_request",
                error_message="请求包含尝试绕过系统规则或披露系统信息的内容。",
            )

        if self._secret_pattern.search(normalized):
            return AgentRequestSecurityDecision(
                allowed=False,
                stop_reason="unsafe_request",
                error_message="不能读取、导出或披露密钥、令牌、密码等敏感信息。",
            )

        if self._destructive_pattern.search(normalized) or (
            self._source_mutation_pattern.search(normalized)
            and self._source_target_pattern.search(normalized)
        ):
            return AgentRequestSecurityDecision(
                allowed=False,
                stop_reason="unsafe_request",
                error_message="当前 Agent 仅支持受控查询与测试，不执行删除或破坏性操作。",
            )

        if self._unregistered_execution_pattern.search(normalized):
            return AgentRequestSecurityDecision(
                allowed=False,
                stop_reason="unsupported_request",
                error_message="当前未注册部署或任意 Shell 执行工具。",
            )

        if self._out_of_scope_pattern.search(normalized):
            return AgentRequestSecurityDecision(
                allowed=False,
                stop_reason="out_of_scope_request",
                error_message="该问题与当前代码仓库理解和研发问答范围无关。",
            )

        return AgentRequestSecurityDecision(allowed=True)
