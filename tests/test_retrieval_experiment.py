from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from evaluation.retrieval_experiment import (
    RetrievalExperimentMethod,
    run_retrieval_experiment,
    summarize_case_reports,
)


def test_run_retrieval_experiment_summarizes_methods():
    cases = [
        {
            "case_id": "case-1",
            "query": "where is keyword_score",
            "expected_chunk_ids": ["expected"],
        }
    ]
    methods = [
        RetrievalExperimentMethod(
            name="fake",
            runner=lambda query: {
                "results": [
                    {
                        "chunk_id": "expected",
                        "content": query,
                    }
                ]
            },
        )
    ]

    report = run_retrieval_experiment(
        cases=cases,
        methods=methods,
        top_k=1,
    )

    assert report["case_count"] == 1
    assert report["method_count"] == 1
    assert report["methods"][0]["method"] == "fake"
    assert report["methods"][0]["summary"]["hit_at_k"] == 1.0
    assert report["methods"][0]["summary"]["mrr"] == 1.0


def test_summarize_case_reports_handles_empty_list():
    summary = summarize_case_reports([])

    assert summary["hit_at_k"] == 0.0
    assert summary["average_latency_ms"] == 0.0
