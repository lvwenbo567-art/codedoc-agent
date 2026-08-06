from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
sys.path.insert(0, str(APP_DIR))

from evaluation.retrieval_experiment import (  # noqa: E402
    build_retrieval_experiment_methods,
    ensure_mock_vector_index,
    load_experiment_cases,
    run_retrieval_experiment,
    save_retrieval_experiment_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval ablation experiments for CodeDoc.",
    )
    parser.add_argument(
        "--dataset",
        default="data/evaluation/retrieval_experiment_cases.jsonl",
    )
    parser.add_argument(
        "--chunks-path",
        default="outputs/test_project_chunks.json",
    )
    parser.add_argument(
        "--index-path",
        default="outputs/experiments/test_project_vector_index_mock.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/experiments/retrieval_experiment_report.json",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--embedding-provider", default="mock")
    parser.add_argument("--embedding-model", default="mock-hash-embedding")
    parser.add_argument("--embedding-base-url", default="http://localhost:11434")
    parser.add_argument("--mock-dimension", type=int, default=64)
    parser.add_argument("--rerank-provider", default="mock")
    parser.add_argument("--rerank-model", default="mock-reranker")
    parser.add_argument("--query-rewrite-provider", default="mock")
    parser.add_argument("--query-rewrite-model", default="mock-chat-model")
    parser.add_argument(
        "--prepare-mock-index",
        action="store_true",
        help="Build a fresh mock vector index before running experiments.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prepare_mock_index:
        ensure_mock_vector_index(
            chunks_path=args.chunks_path,
            index_path=args.index_path,
            embedding_model=args.embedding_model,
            mock_dimension=args.mock_dimension,
        )

    cases = load_experiment_cases(args.dataset)
    methods = build_retrieval_experiment_methods(
        chunks_path=args.chunks_path,
        index_path=args.index_path,
        top_k=args.top_k,
        candidate_top_k=args.candidate_top_k,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        mock_dimension=args.mock_dimension,
        rerank_provider=args.rerank_provider,
        rerank_model=args.rerank_model,
        query_rewrite_provider=args.query_rewrite_provider,
        query_rewrite_model=args.query_rewrite_model,
    )
    report = run_retrieval_experiment(
        cases=cases,
        methods=methods,
        top_k=args.top_k,
    )
    output_path = save_retrieval_experiment_report(
        report=report,
        output_path=args.output,
    )

    print(
        {
            "output_path": output_path,
            "case_count": report["case_count"],
            "method_count": report["method_count"],
        }
    )

    for method in report["methods"]:
        print(method["method"], method["summary"])


if __name__ == "__main__":
    main()
