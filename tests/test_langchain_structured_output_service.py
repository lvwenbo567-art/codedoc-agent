from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from langchain_core.messages import AIMessage

from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.output_schema import QueryAnalysis
from langchain_agent.structured_output_service import (
    QueryAnalysisService,
    extract_protected_terms,
)


class FakeStructuredRunnable:
    def invoke(self, messages):
        return {
            "raw": AIMessage(
                content='{"query_type":"code"}'
            ),
            "parsed": QueryAnalysis(
                query_type="code",
                recommended_tool="search_code",
                recommended_query_strategy="original",
                needs_rewrite=False,
                protected_terms=[
                    "RerankClient.score",
                ],
                classification_reason="包含明确方法名。",
            ),
            "parsing_error": None,
        }


class FakeStructuredModel:
    def __init__(self):
        self.schema = None
        self.method = None
        self.include_raw = None

    def with_structured_output(
        self,
        schema,
        *,
        method,
        include_raw,
    ):
        self.schema = schema
        self.method = method
        self.include_raw = include_raw

        return FakeStructuredRunnable()


def test_structured_output_success():
    model = FakeStructuredModel()

    config = LangChainModelConfig(
        provider="openai_compatible",
        model_name="fake-model",
        base_url="http://localhost/v1",
    )

    service = QueryAnalysisService(
        config=config,
        model=model,
    )

    result = service.analyze(
        "RerankClient.score 在哪里实现？"
    )

    assert result.mode == "structured_output"
    assert result.analysis.query_type == "code"
    assert result.analysis.recommended_tool == "search_code"
    assert model.schema is QueryAnalysis
    assert model.method == "function_calling"
    assert model.include_raw is True


def test_mock_rules_classify_structure():
    service = QueryAnalysisService(
        config=LangChainModelConfig(provider="mock")
    )

    result = service.analyze(
        "这个项目有哪些主要目录和模块？"
    )

    assert result.analysis.query_type == "structure"
    assert result.analysis.recommended_tool == "get_project_structure"


def test_mock_rules_preserve_identifier():
    service = QueryAnalysisService(
        config=LangChainModelConfig(provider="mock")
    )

    result = service.analyze(
        "EmbeddingClient 是如何生成向量的？"
    ).analysis

    assert result.query_type == "code"
    assert "EmbeddingClient" in result.protected_terms
    assert result.recommended_query_strategy == "original"


def test_mock_rules_use_multi_query_for_document_without_protected_terms():
    service = QueryAnalysisService(
        config=LangChainModelConfig(provider="mock")
    )

    result = service.analyze(
        "如何启动这个项目？"
    ).analysis

    assert result.query_type == "document"
    assert result.recommended_tool == "search_documents"
    assert result.recommended_query_strategy == "multi_query"
    assert result.needs_rewrite is True


def test_mock_rules_classify_code_and_document_as_mixed():
    service = QueryAnalysisService(
        config=LangChainModelConfig(provider="mock")
    )

    result = service.analyze(
        "README 里有没有说明 keyword_score 函数怎么使用？"
    ).analysis

    assert result.query_type == "mixed"
    assert result.recommended_tool == "multiple"
    assert result.recommended_query_strategy == "original"


def test_extract_protected_terms_keeps_api_path_and_method():
    terms = extract_protected_terms(
        "POST /ask 和 RerankClient.score 有什么区别？"
    )

    assert "/ask" in terms
    assert "RerankClient" in terms


class BrokenStructuredModel:
    def with_structured_output(
        self,
        schema,
        *,
        method,
        include_raw,
    ):
        raise RuntimeError(
            "model does not support structured output"
        )


def test_structured_output_failure_falls_back():
    config = LangChainModelConfig(
        provider="openai_compatible",
        model_name="fake-model",
        base_url="http://localhost/v1",
    )

    service = QueryAnalysisService(
        config=config,
        model=BrokenStructuredModel(),
    )

    result = service.analyze(
        "如何启动这个项目？"
    )

    assert result.fallback_used is True
    assert result.mode == "rule_fallback"
    assert result.analysis.query_type == "document"

