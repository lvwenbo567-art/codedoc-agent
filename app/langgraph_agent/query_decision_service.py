from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.model_factory import create_chat_model
from langgraph_agent.decision_schema import QueryDecision
from langgraph_agent.query_classifier import RuleBasedQueryClassifier


BACKTICK_SYMBOL_PATTERN = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)`")#识别代码符号keyword_score
DOTTED_SYMBOL_PATTERN = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b"
)#匹配带点的符号RerankClient.score
SNAKE_CASE_PATTERN = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]*)\b"
)#匹配 snake_case
CAMEL_CASE_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9]*(?:Client|Service|Config|Registry|Executor|State|Graph|Middleware|Node))\b"
)#匹配常见类名RerankClient


QUERY_DECISION_SYSTEM_PROMPT = """
你是 CodeDoc Query Router。

请判断用户问题属于 code、document、structure、unknown 哪一类，
并选择 original、multi_query、structure、none 检索策略。

如果问题包含明确函数名、类名、方法名、文件名或 API 路径，优先 original。
如果问题是自然语言描述的代码/文档问题，优先 multi_query。
如果问题询问项目目录、模块、入口文件，使用 structure。
如果问题与当前代码项目无关，使用 none。

symbol_name 只有在能明确识别代码标识符时才填写，不能猜测。
""".strip()

CODE_SEMANTIC_KEYWORDS = (
    "召回",
    "重排",
    "rerank",
    "multi-query",
    "multi_query",
    "hybrid",
    "pipeline",
    "检索流程",
)


CODE_SEMANTIC_ASCII_KEYWORDS = (
    "多路",
    "召回",
    "重排",
    "检索流程",
    "rerank",
    "multi-query",
    "multi_query",
    "hybrid",
    "pipeline",
    "retrieval",
    "retrieve",
    "search",
)


class QueryDecisionService:
    def __init__(
        self,
        *,
        model_config: LangChainModelConfig,
    ) -> None:
        self.model_config = model_config
        self.rule_classifier = RuleBasedQueryClassifier()
        self._structured_model = None

        if model_config.provider != "mock":
            model = create_chat_model(model_config)
            self._structured_model = model.with_structured_output(QueryDecision)

    @staticmethod
    def extract_symbol_candidate(query: str) -> str | None:
        for pattern in (
            BACKTICK_SYMBOL_PATTERN,
            DOTTED_SYMBOL_PATTERN,
            SNAKE_CASE_PATTERN,
            CAMEL_CASE_PATTERN,
        ):
            '''
            优先级是：
1. 反引号里的符号
2. 带点符号
3. snake_case
4. CamelCase 类名
            '''
            match = pattern.search(query)

            if match:
                return match.group(1)

        return None

    @staticmethod
    def _is_code_semantic_query(query: str) -> bool:
        normalized_query = query.lower()
        keyword_groups = (
            CODE_SEMANTIC_KEYWORDS,
            CODE_SEMANTIC_ASCII_KEYWORDS,
        )

        return any(
            keyword.lower() in normalized_query
            for keywords in keyword_groups
            for keyword in keywords
        )

    def _build_rule_decision(
        self,
        query: str,
        *,
        decision_method: str,
    ) -> QueryDecision:
        '''
        这是规则决策函数。
它根据规则直接构造一个 QueryDecision。
        '''
        symbol_name = self.extract_symbol_candidate(query)
        query_type = self.rule_classifier.classify(query)

        if query_type == "unknown" and symbol_name:
            query_type = "code"

        if (
            query_type == "unknown"
            and self._is_code_semantic_query(query)
        ):
            query_type = "code"

        if query_type != "code":
            symbol_name = None

        if query_type == "structure":
            strategy = "structure"
        elif query_type == "code":
            strategy = "original" if symbol_name else "multi_query"
        elif query_type == "document":
            strategy = "multi_query"
        else:
            strategy = "none"

        return QueryDecision(
            query_type=query_type,
            retrieval_strategy=strategy,
            symbol_name=symbol_name,
            confidence=0.85 if query_type != "unknown" else 0.35,
            reason="根据关键词和代码标识符规则完成分类",
            decision_method=decision_method,
        )

    def analyze(self, query: str) -> QueryDecision:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query 不能为空")

        rule_decision = self._build_rule_decision(
            normalized_query,
            decision_method="rule",
        )

        if self._structured_model is None:
            return rule_decision

        try:
            decision = self._structured_model.invoke(
                [
                    SystemMessage(content=QUERY_DECISION_SYSTEM_PROMPT),
                    HumanMessage(content=f"用户问题：\n{normalized_query}"),
                ]
            )

            if not isinstance(decision, QueryDecision):
                decision = QueryDecision.model_validate(decision)

            symbol_name = (
                decision.symbol_name
                or self.extract_symbol_candidate(normalized_query)
            )
            query_type = decision.query_type
            strategy = decision.retrieval_strategy

            if self._is_code_semantic_query(
                normalized_query
            ):
                query_type = "code"

            if query_type == "structure":
                strategy = "structure"
            elif query_type == "unknown":
                strategy = "none"
            elif query_type == "code":
                strategy = (
                    "original"
                    if symbol_name
                    else "multi_query"
                )

            return decision.model_copy(
                update={
                    "query_type": query_type,
                    "symbol_name": symbol_name,
                    "retrieval_strategy": strategy,
                    "decision_method": "model",
                }
            )

        except Exception:
            return self._build_rule_decision(
                normalized_query,
                decision_method="rule_fallback",
            )
