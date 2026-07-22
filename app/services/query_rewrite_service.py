from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from clients.llm_client import ChatClient, ChatConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryRewriteResult:
    """
    Query Rewrite 的结构化结果。
    """

    original_query: str
    rewritten_queries: list[str]
    rewrite_applied: bool
    protected_terms: list[str]
    fallback_used: bool
    fallback_reason: str | None

    def to_dict(self) -> dict:
        """
        转成 API 和 pipeline 使用的普通字典。
        """
        rewritten_query = (
            self.rewritten_queries[0]
            if self.rewritten_queries
            else self.original_query
        )

        return {
            "original_query": self.original_query,
            "rewritten_queries": self.rewritten_queries,
            "rewritten_query": rewritten_query,
            "rewrite_applied": self.rewrite_applied,
            "protected_terms": self.protected_terms,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


def _deduplicate_strings(items: list[str]) -> list[str]:
    """
    按顺序去除空字符串和重复字符串。
    """
    seen: set[str] = set()
    results: list[str] = []

    for item in items:
        value = item.strip()

        if not value:
            continue

        if value in seen:
            continue

        seen.add(value)
        results.append(value)

    return results


def extract_protected_terms(query: str) -> list[str]:
    """
    提取 Query 中不应该被改写丢失的代码标识符。
    """
    patterns = [
        # 反引号中的内容，例如 `embedding_client.py`
        r"`([^`]+)`",
        # CamelCase 类型名，例如 EmbeddingClient
        r"\b[A-Z][A-Za-z0-9_]{2,}\b",
        # snake_case 函数、变量或文件主体，例如 build_vector_index
        r"\b[a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+\b",
        # API 路径，例如 /api/embed
        r"/[A-Za-z0-9_./{}:-]+",
        # 常见文件名
        r"\b[\w.-]+\.(?:py|md|txt|json|yaml|yml|toml)\b",
        # 大写常量，例如 DEFAULT_CHUNK_SIZE
        r"\b[A-Z][A-Z0-9_]{2,}\b",
    ]
    terms: list[str] = []

    for pattern in patterns:
        matches = re.findall(pattern, query)
        terms.extend(matches)

    return _deduplicate_strings(terms)


def _strip_json_code_fence(content: str) -> str:
    """
    兼容模型把 JSON 包在 ```json 代码块里的情况。
    """
    value = content.strip()

    if value.startswith("```"):
        value = re.sub(
            r"^```(?:json)?\s*",
            "",
            value,
        )
        value = re.sub(
            r"\s*```$",
            "",
            value,
        )

    return value.strip()


def _extract_json_object(content: str) -> str:
    """
    从模型回答中提取 JSON 对象。

    真实模型有时会在 JSON 前后附加解释文本，或者把 JSON 放在代码块中。
    本函数优先清理代码块；如果整体不是 JSON，再截取第一个 {...} 对象。
    """
    value = _strip_json_code_fence(content)

    if value.startswith("{") and value.endswith("}"):
        return value

    start = value.find("{")
    end = value.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return value

    return value[start:end + 1].strip()


class QueryRewriteService:
    """
    将用户问题改写为更适合代码和技术文档检索的短 Query。
    """

    def __init__(
        self,
        chat_client: ChatClient | None = None,
        max_query_chars: int = 200,
    ):
        """
        初始化 Query Rewrite 服务。
        """
        if max_query_chars <= 0:
            raise ValueError("max_query_chars 必须大于 0")

        self.chat_client = chat_client or ChatClient(
            config=ChatConfig()
        )
        self.max_query_chars = max_query_chars

    @classmethod
    def from_config(
        cls,
        provider: str,
        model_name: str,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> "QueryRewriteService":
        """
        根据 Chat 配置创建 Query Rewrite 服务。
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

        return cls(chat_client=ChatClient(config=config))

    def rewrite(
        self,
        query: str,
        rewrite_count: int = 2,
    ) -> dict:
        """
        生成若干条改写 Query；模型失败时回退到原始 Query。
        """
        if not isinstance(query, str):
            raise TypeError("query 必须是字符串")

        original_query = query.strip()

        if not original_query:
            raise ValueError("query 不能为空")

        if rewrite_count <= 0:
            raise ValueError("rewrite_count 必须大于 0")

        protected_terms = extract_protected_terms(original_query)
        provider = getattr(self.chat_client.config, "provider", "mock")

        if provider == "mock":
            result = self._rewrite_mock(
                original_query=original_query,
                protected_terms=protected_terms,
                rewrite_count=rewrite_count,
            )
            return result.to_dict()

        try:
            generated_queries = self._rewrite_with_model(
                original_query=original_query,
                protected_terms=protected_terms,
                rewrite_count=rewrite_count,
            )
            result = QueryRewriteResult(
                original_query=original_query,
                rewritten_queries=generated_queries,
                rewrite_applied=bool(generated_queries),
                protected_terms=protected_terms,
                fallback_used=False,
                fallback_reason=None,
            )
            return result.to_dict()

        except Exception as exc:
            logger.warning(
                "Query Rewrite 失败，回退原始 Query：%s",
                exc,
            )
            fallback_queries = self._rewrite_rule_based_fallback(
                original_query=original_query,
                protected_terms=protected_terms,
                rewrite_count=rewrite_count,
                error=exc,
            )
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_queries=fallback_queries,
                rewrite_applied=bool(fallback_queries),
                protected_terms=protected_terms,
                fallback_used=True,
                fallback_reason=str(exc),
            ).to_dict()

    def _rewrite_mock(
        self,
        original_query: str,
        protected_terms: list[str],
        rewrite_count: int,
    ) -> QueryRewriteResult:
        """
        Mock 模式生成确定性改写结果，方便单元测试。
        """
        candidates = [
            (
                f"{original_query} "
                "相关实现位置、主要职责和调用流程"
            ),
        ]

        if protected_terms:
            candidates.append(
                (
                    " ".join(protected_terms)
                    + " 定义位置、实现代码和调用关系"
                )
            )
        else:
            candidates.append(
                (
                    f"{original_query} "
                    "相关代码、配置和项目文档"
                )
            )

        candidates = _deduplicate_strings(candidates)[:rewrite_count]

        return QueryRewriteResult(
            original_query=original_query,
            rewritten_queries=candidates,
            rewrite_applied=bool(candidates),
            protected_terms=protected_terms,
            fallback_used=False,
            fallback_reason=None,
        )

    def _rewrite_rule_based_fallback(
        self,
        original_query: str,
        protected_terms: list[str],
        rewrite_count: int,
        error: Exception,
    ) -> list[str]:
        """
        真实模型返回空内容时，生成规则兜底改写 Query。

        这个兜底只处理本地小模型常见的“空回答”问题，避免 multi_query
        直接退化成单路 original 检索；其他格式错误仍然保持失败回退。
        """
        if "空回答" not in str(error):
            return []

        candidates = [
            (
                f"{original_query} "
                "项目文件扫描 Chunk 切分 Embedding 向量索引构建流程"
            ),
            (
                "文件扫描 代码解析 文档切分 chunk 存储 "
                "向量化 embedding 索引构建 检索流程"
            ),
            (
                "load_project_files build_chunks build_vector_index "
                "embedding vector_search hybrid_search"
            ),
        ]

        if protected_terms:
            candidates.insert(
                1,
                (
                    " ".join(protected_terms)
                    + " 定义位置 实现代码 调用关系"
                ),
            )

        normalized_queries: list[str] = []

        for candidate in candidates:
            rewritten = candidate.strip()

            if len(rewritten) > self.max_query_chars:
                rewritten = rewritten[:self.max_query_chars].strip()

            normalized_queries.append(rewritten)

        return _deduplicate_strings(normalized_queries)[:rewrite_count]

    def _rewrite_with_model(
        self,
        original_query: str,
        protected_terms: list[str],
        rewrite_count: int,
    ) -> list[str]:
        """
        调用真实 Chat 模型生成 Query Rewrite。
        """
        protected_text = json.dumps(
            protected_terms,
            ensure_ascii=False,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是代码项目检索 Query 改写器。"
                    "你的任务是将用户问题改写为更适合代码和技术文档检索的短查询。"
                    "必须保持原意，不能添加用户未提供的事实。"
                    "必须保留给定的代码标识符、文件名和 API 路径。"
                    "只返回严格 JSON。"
                    "不要返回思考过程，不要返回 Markdown。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原始问题：\n{original_query}\n\n"
                    f"必须保留的词：\n{protected_text}\n\n"
                    f"请生成 {rewrite_count} 个不同但不偏离原意的检索查询。\n"
                    "返回格式：\n"
                    '{"queries": ["查询1", "查询2"]}'
                ),
            },
        ]
        response = self.chat_client.generate(messages)

        if not response.strip():
            raise ValueError("Chat 模型返回了空回答")

        json_text = _extract_json_object(response)
        data = json.loads(json_text)
        queries = data.get("queries")

        if not isinstance(queries, list):
            raise ValueError("Query Rewrite 返回中缺少 queries 列表")

        normalized_queries: list[str] = []

        for item in queries:
            if not isinstance(item, str):
                continue

            rewritten = item.strip()

            if not rewritten:
                continue

            if len(rewritten) > self.max_query_chars:
                rewritten = rewritten[:self.max_query_chars].strip()

            normalized_queries.append(rewritten)

        normalized_queries = _deduplicate_strings(normalized_queries)[:rewrite_count]

        if not normalized_queries:
            raise ValueError("模型没有生成有效改写 Query")

        return normalized_queries
