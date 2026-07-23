from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain_agent.chat_service import extract_message_text
from langchain_agent.message_builder import build_query_analysis_messages
from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.model_factory import create_chat_model
from langchain_agent.output_schema import QueryAnalysis, QueryAnalysisResult
from services.query_rewrite_service import extract_protected_terms


class QueryAnalysisService:
    """
    将自由文本问题转换为经过 Pydantic 校验的 QueryAnalysis。

    这一层用于 Day31 的 LangChain Structured Output，也会成为后续
    LangGraph 检索路由的前置判断入口。
    """

    STRUCTURE_KEYWORDS = {
        "目录",
        "项目结构",
        "模块结构",
        "有哪些文件",
        "入口文件",
        "文件组织",
        "目录树",
        "project structure",
    }

    CODE_KEYWORDS = {
        "函数",
        "方法",
        "类",
        "代码",
        "实现",
        "调用",
        "参数",
        "返回值",
        "源码",
        "client",
        "service",
        "score",
    }

    DOCUMENT_KEYWORDS = {
        "README",
        "readme",
        "文档",
        "启动方式",
        "启动",
        "安装",
        "部署说明",
        "部署",
        "设计说明",
        "配置说明",
        "配置",
        "使用方法",
        "使用",
    }

    def __init__(
        self,
        *,
        config: LangChainModelConfig,
        model: BaseChatModel | Any | None = None,
    ) -> None:
        self.config = config
        self._model = model

    def _get_model(self) -> BaseChatModel:
        """
        懒加载真实 ChatModel。
        """
        if self._model is None:
            self._model = create_chat_model(self.config)

        return self._model

    def analyze(self, query: str) -> QueryAnalysisResult:
        """
        分析用户问题的类型、推荐工具和推荐查询策略。
        """
        query = query.strip()

        if not query:
            raise ValueError("query 不能为空")

        start_time = time.perf_counter()

        if self.config.provider == "mock" and self._model is None:
            return QueryAnalysisResult(
                query=query,
                analysis=self._analyze_by_rules(query),
                provider=self.config.provider,
                model_name=self.config.model_name,
                mode="mock_rules",
                fallback_used=False,
                duration_ms=self._duration_ms(start_time),
            )

        try:
            analysis, raw_content = self._analyze_with_model(query)

            return QueryAnalysisResult(
                query=query,
                analysis=analysis,
                provider=self.config.provider,
                model_name=self.config.model_name,
                mode="structured_output",
                fallback_used=False,
                raw_content=raw_content,
                duration_ms=self._duration_ms(start_time),
            )

        except Exception as exc:
            fallback_analysis = self._analyze_by_rules(query)

            return QueryAnalysisResult(
                query=query,
                analysis=fallback_analysis,
                provider=self.config.provider,
                model_name=self.config.model_name,
                mode="rule_fallback",
                fallback_used=True,
                error_message=str(exc),
                duration_ms=self._duration_ms(start_time),
            )

    def _analyze_with_model(self, query: str) -> tuple[QueryAnalysis, str | None]:
        """
        调用真实 LangChain Structured Output。

        返回：
        - QueryAnalysis：Pydantic 校验后的结构化结果；
        - raw_content：模型原始文本，便于排查结构化输出问题。
        """
        model = self._get_model()

        structured_model = model.with_structured_output(
            QueryAnalysis,
            method=self.config.structured_output_method,
            include_raw=True,
        )

        messages = build_query_analysis_messages(query)
        result = structured_model.invoke(messages)

        # include_raw=True 的标准结果通常是：
        # {"raw": AIMessage, "parsed": QueryAnalysis | dict, "parsing_error": ...}
        if isinstance(result, dict):
            parsing_error = result.get("parsing_error")

            if parsing_error is not None:
                raise ValueError(f"结构化输出解析失败：{parsing_error}")

            parsed = result.get("parsed")
            raw = result.get("raw")

        else:
            parsed = result
            raw = None

        if parsed is None:
            raise ValueError("模型未返回 parsed 结果")

        if isinstance(parsed, QueryAnalysis):
            analysis = parsed

        elif isinstance(parsed, dict):
            analysis = QueryAnalysis.model_validate(parsed)

        else:
            raise TypeError(
                "无法识别的结构化输出类型："
                f"{type(parsed).__name__}"
            )

        raw_content: str | None = None

        if isinstance(raw, AIMessage):
            raw_content = extract_message_text(raw)

        elif raw is not None:
            raw_content = str(raw)

        return analysis, raw_content

    def _analyze_by_rules(self, query: str) -> QueryAnalysis:
        """
        用本地规则完成 Query 分类。

        用途：
        1. mock 模式下不调用真实模型；
        2. 真实模型结构化输出失败时作为降级方案。
        """
        protected_terms = extract_protected_terms(query)
        lowered_query = query.lower()

        has_structure = any(
            keyword.lower() in lowered_query
            for keyword in self.STRUCTURE_KEYWORDS
        )
        has_code_keyword = any(
            keyword.lower() in lowered_query
            for keyword in self.CODE_KEYWORDS
        )
        has_document = any(
            keyword.lower() in lowered_query
            for keyword in self.DOCUMENT_KEYWORDS
        )
        has_code_identifier = bool(protected_terms) or bool(
            re.search(
                r"\b[A-Za-z_][A-Za-z0-9_]*"
                r"\.[A-Za-z_][A-Za-z0-9_]*\b",
                query,
            )
        )
        has_code = has_code_keyword or has_code_identifier

        if has_structure and not has_code and not has_document:
            query_type = "structure"
            recommended_tool = "get_project_structure"

        elif has_code and has_document:
            query_type = "mixed"
            recommended_tool = "multiple"

        elif has_code:
            query_type = "code"
            recommended_tool = "search_code"

        elif has_document:
            query_type = "document"
            recommended_tool = "search_documents"

        else:
            query_type = "unknown"
            recommended_tool = "none"

        if query_type == "structure" or protected_terms:
            query_strategy = "original"
            needs_rewrite = False

        elif query_type in {
            "code",
            "document",
            "mixed",
        }:
            query_strategy = "multi_query"
            needs_rewrite = True

        else:
            query_strategy = "original"
            needs_rewrite = False

        if protected_terms:
            reason = (
                "问题包含明确代码标识符："
                + "、".join(protected_terms[:5])
            )

        elif query_type == "structure":
            reason = "问题主要询问项目目录和模块结构。"

        elif query_type == "code":
            reason = "问题主要询问代码实现或调用关系。"

        elif query_type == "document":
            reason = "问题主要询问项目文档和使用说明。"

        elif query_type == "mixed":
            reason = "问题同时涉及代码实现和项目文档。"

        else:
            reason = "缺少足够特征，暂时无法可靠分类。"

        return QueryAnalysis(
            query_type=query_type,
            recommended_tool=recommended_tool,
            recommended_query_strategy=query_strategy,
            needs_rewrite=needs_rewrite,
            protected_terms=protected_terms,
            classification_reason=reason,
        )

    @staticmethod
    def _duration_ms(start_time: float) -> float:
        """
        计算分析耗时，单位毫秒。
        """
        return round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

