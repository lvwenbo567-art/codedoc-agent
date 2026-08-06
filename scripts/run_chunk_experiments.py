from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
sys.path.insert(0, str(APP_DIR))

from evaluation.chunk_experiment import (  # noqa: E402
    load_chunk_experiment_cases,
    load_chunks,
    run_chunk_experiment,
    save_chunk_experiment_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chunk quality experiments for CodeDoc.",
    )
    parser.add_argument(
        "--dataset",
        default="data/evaluation/chunk_experiment_cases.jsonl",
    )
    parser.add_argument(
        "--chunks-path",
        default="outputs/test_project_chunks.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/experiments/chunk_experiment_report.json",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_chunk_experiment_cases(args.dataset)
    chunks = load_chunks(args.chunks_path)
    report = run_chunk_experiment(
        cases=cases,
        chunks=chunks,
    )
    output_path = save_chunk_experiment_report(
        report=report,
        output_path=args.output,
    )

    print(
        {
            "output_path": output_path,
            "case_count": report["case_count"],
            "chunk_count": report["chunk_count"],
            "summary": report["summary"],
        }
    )


if __name__ == "__main__":
    main()
