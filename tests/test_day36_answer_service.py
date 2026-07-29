from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.answer_service import (
    GraphAnswerResult,
    GraphAnswerService,
)


def test_graph_answer_service_generates_with_mock_model():
    service = GraphAnswerService(
        model_config=LangChainModelConfig(provider="mock"),
    )

    result = service.generate(
        query="keyword_score 在哪里？",
        evidence=[
            {
                "chunk_id": "test_project/search.py::keyword_score::part_0",
                "source_path": "test_project/search.py",
                "source_name": "search.py",
                "chunk_type": "code",
                "evidence_type": "code_search",
                "content": "def keyword_score(query, text): return 1",
                "score": 0.9,
            }
        ],
        max_context_chars=1000,
    )

    assert isinstance(result, GraphAnswerResult)
    assert "[Source 1]" in result.answer
    assert result.citations[0]["citation_id"] == "Source 1"
    assert result.answer_quality["is_valid"] is True
    assert result.chunks[0]["source_path"] == "test_project/search.py"
