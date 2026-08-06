from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
)
from evaluation.retrieval_metrics import (
    calculate_hit_at_k,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)
from pipelines.retrieval_pipeline import retrieve_with_rerank
from services.hybrid_search_service import hybrid_search_from_files
from services.index_service import build_vector_index_from_json
from services.keyword_search_service import search_chunks_from_json
from services.vector_search_gateway import search_vector_store


@dataclass(frozen=True)
class RetrievalExperimentMethod:
    name: str
    runner: Callable[[str], dict[str, Any]]


def load_experiment_cases(path: str) -> list[dict[str, Any]]:
    """
    加载 JSONL 检索实验数据集。
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"实验数据集不存在：{dataset_path}")

    cases: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = line.strip()

        if not value:
            continue

        try:
            item = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"实验数据集第 {line_number} 行不是合法 JSON"
            ) from exc

        if not item.get("query"):
            raise ValueError(f"实验数据集第 {line_number} 行缺少 query")

        if not item.get("expected_chunk_ids"):
            raise ValueError(
                f"实验数据集第 {line_number} 行缺少 expected_chunk_ids"
            )

        cases.append(item)

    return cases


def ensure_mock_vector_index(
    *,
    chunks_path: str,
    index_path: str,
    embedding_model: str = "mock-hash-embedding",
    mock_dimension: int = 64,
) -> dict[str, Any]:
    """
    为离线实验构建稳定的 mock 向量索引。
    """
    return build_vector_index_from_json(
        chunks_path=chunks_path,
        output_path=index_path,
        embedding_provider="mock",
        embedding_model=embedding_model,
        mock_dimension=mock_dimension,
        batch_size=16,
        incremental=False,
    )


def build_retrieval_experiment_methods(
    *,
    chunks_path: str,
    index_path: str,
    top_k: int,
    candidate_top_k: int,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    mock_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    rerank_provider: str = "mock",
    rerank_model: str = "mock-reranker",
    query_rewrite_provider: str = "mock",
    query_rewrite_model: str = "mock-chat-model",
) -> list[RetrievalExperimentMethod]:
    """
    构建需要对比的检索策略。
    """

    def run_bm25(query: str) -> dict[str, Any]:
        results = search_chunks_from_json(
            input_path=chunks_path,
            query=query,
            top_k=top_k,
        )

        return {"results": results}

    def run_vector(query: str) -> dict[str, Any]:
        return search_vector_store(
            query=query,
            project_id=1,
            index_path=index_path,
            top_k=top_k,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            mock_dimension=mock_dimension,
            include_content=True,
        )

    def run_hybrid(query: str) -> dict[str, Any]:
        return hybrid_search_from_files(
            query=query,
            chunks_path=chunks_path,
            index_path=index_path,
            keyword_top_k=candidate_top_k,
            vector_top_k=candidate_top_k,
            final_top_k=top_k,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            mock_dimension=mock_dimension,
        )

    def run_hybrid_rerank(query: str) -> dict[str, Any]:
        return retrieve_with_rerank(
            query=query,
            chunks_path=chunks_path,
            index_path=index_path,
            candidate_top_k=candidate_top_k,
            final_top_k=top_k,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            mock_dimension=mock_dimension,
            rerank_provider=rerank_provider,
            rerank_model=rerank_model,
            query_strategy="original",
        )

    def run_multi_query_rerank(query: str) -> dict[str, Any]:
        return retrieve_with_rerank(
            query=query,
            chunks_path=chunks_path,
            index_path=index_path,
            candidate_top_k=candidate_top_k,
            final_top_k=top_k,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            mock_dimension=mock_dimension,
            rerank_provider=rerank_provider,
            rerank_model=rerank_model,
            query_strategy="multi_query",
            query_rewrite_provider=query_rewrite_provider,
            query_rewrite_model=query_rewrite_model,
        )

    return [
        RetrievalExperimentMethod("bm25", run_bm25),
        RetrievalExperimentMethod("vector", run_vector),
        RetrievalExperimentMethod("hybrid", run_hybrid),
        RetrievalExperimentMethod("hybrid_rerank", run_hybrid_rerank),
        RetrievalExperimentMethod("multi_query_rerank", run_multi_query_rerank),
    ]


def evaluate_retrieval_results(
    *,
    results: list[dict[str, Any]],
    expected_chunk_ids: list[str],
    top_k: int,
) -> dict[str, float]:
    return {
        "hit_at_k": calculate_hit_at_k(
            results,
            expected_chunk_ids,
            top_k,
        ),
        "recall_at_k": calculate_recall_at_k(
            results,
            expected_chunk_ids,
            top_k,
        ),
        "mrr": calculate_reciprocal_rank(
            results,
            expected_chunk_ids,
        ),
        "ndcg_at_k": calculate_ndcg_at_k(
            results,
            expected_chunk_ids,
            top_k,
        ),
    }


def run_retrieval_experiment(
    *,
    cases: list[dict[str, Any]],
    methods: list[RetrievalExperimentMethod],
    top_k: int,
) -> dict[str, Any]:
    """
    对同一批 query 执行多种检索策略并生成消融报告。
    """
    method_reports: list[dict[str, Any]] = []

    for method in methods:
        case_reports: list[dict[str, Any]] = []

        for case in cases:
            started_at = time.perf_counter()
            output = method.runner(str(case["query"]))
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            results = list(output.get("results") or [])
            metrics = evaluate_retrieval_results(
                results=results,
                expected_chunk_ids=list(case["expected_chunk_ids"]),
                top_k=top_k,
            )
            case_reports.append(
                {
                    "case_id": case.get("case_id"),
                    "query": case["query"],
                    "expected_chunk_ids": case["expected_chunk_ids"],
                    "retrieved_chunk_ids": [
                        item.get("chunk_id")
                        for item in results[:top_k]
                    ],
                    "latency_ms": latency_ms,
                    **metrics,
                }
            )

        method_reports.append(
            {
                "method": method.name,
                "case_count": len(case_reports),
                "summary": summarize_case_reports(case_reports),
                "cases": case_reports,
            }
        )

    return {
        "top_k": top_k,
        "method_count": len(method_reports),
        "case_count": len(cases),
        "methods": method_reports,
    }


def summarize_case_reports(
    case_reports: list[dict[str, Any]],
) -> dict[str, float]:
    if not case_reports:
        return {
            "hit_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "average_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }

    latencies = sorted(float(item["latency_ms"]) for item in case_reports)
    p95_index = max(0, min(len(latencies) - 1, round(len(latencies) * 0.95) - 1))

    return {
        "hit_at_k": _average(case_reports, "hit_at_k"),
        "recall_at_k": _average(case_reports, "recall_at_k"),
        "mrr": _average(case_reports, "mrr"),
        "ndcg_at_k": _average(case_reports, "ndcg_at_k"),
        "average_latency_ms": sum(latencies) / len(latencies),
        "p95_latency_ms": latencies[p95_index],
    }


def save_retrieval_experiment_report(
    *,
    report: dict[str, Any],
    output_path: str,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return str(path)


def _average(
    items: list[dict[str, Any]],
    key: str,
) -> float:
    return sum(float(item[key]) for item in items) / len(items)
