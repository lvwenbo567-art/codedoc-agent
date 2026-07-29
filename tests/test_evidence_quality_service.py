from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.evidence_quality_service import EvidenceQualityService


def build_service() -> EvidenceQualityService:
    return EvidenceQualityService(
        model_config=LangChainModelConfig(provider="mock")
    )


def test_model_exception_falls_back_to_rules():
    class BrokenStructuredModel:
        def invoke(self, messages):
            raise RuntimeError("model down")

    service = build_service()
    service._structured_model = BrokenStructuredModel()

    result = service.assess(
        query="keyword_score 在哪里？",
        query_type="code",
        symbol_name="keyword_score",
        evidence=[
            {
                "source_path": "test_project/search.py",
                "evidence_type": "symbol_lookup",
                "symbol_name": "keyword_score",
                "content": "def keyword_score(query, text): return 1",
            }
        ],
    )

    assert result.assessment_method == "rule_fallback"
    assert result.sufficient is True


def test_empty_evidence_is_insufficient():
    result = build_service().assess(
        query="keyword_score 在哪里？",
        query_type="code",
        evidence=[],
        symbol_name="keyword_score",
    )

    assert result.sufficient is False


def test_exact_symbol_hit_is_sufficient():
    result = build_service().assess(
        query="keyword_score 在哪里？",
        query_type="code",
        symbol_name="keyword_score",
        evidence=[
            {
                "source_path": "test_project/search.py",
                "evidence_type": "symbol_lookup",
                "symbol_name": "keyword_score",
                "content": "def keyword_score(query, text): return 1",
            }
        ],
    )

    assert result.sufficient is True


def test_structure_evidence_is_sufficient():
    result = build_service().assess(
        query="项目有哪些目录？",
        query_type="structure",
        symbol_name=None,
        evidence=[
            {
                "source_path": "<project-structure-summary>",
                "evidence_type": "project_structure",
                "content": "app/api\napp/services",
            }
        ],
    )

    assert result.sufficient is True


def test_short_irrelevant_evidence_is_insufficient():
    result = build_service().assess(
        query="启动方式是什么？",
        query_type="document",
        symbol_name=None,
        evidence=[
            {
                "source_path": "README.md",
                "evidence_type": "document_search",
                "content": "short",
            }
        ],
    )

    assert result.sufficient is False
