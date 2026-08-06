from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
sys.path.append(str(APP_DIR))

from evaluation.agent_eval_dataset import load_agent_eval_cases  # noqa: E402
from evaluation.agent_eval_report import save_agent_eval_report  # noqa: E402
from evaluation.agent_eval_runner import (  # noqa: E402
    ToolAgentEvalExecutor,
    run_agent_evaluation,
)
from langchain_agent.model_config import LangChainModelConfig  # noqa: E402
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig  # noqa: E402
from langgraph_agent.tool_agent_dependencies import (  # noqa: E402
    build_tool_agent_dependencies,
)
from langgraph_agent.tool_agent_service import CodeDocToolAgentService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CodeDoc Agent tool-calling experiments.",
    )
    parser.add_argument(
        "--dataset",
        default="data/evaluation/codedoc_agent_eval.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs/experiments/agent_tool_call_experiment_report.json",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--chunks-path", default="outputs/test_project_chunks.json")
    parser.add_argument(
        "--index-path",
        default="outputs/test_project_vector_index_bge_m3.json",
    )
    parser.add_argument("--embedding-provider", default="ollama")
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--embedding-base-url", default="http://localhost:11434")
    parser.add_argument("--embedding-api-key", default="")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--mock-dimension", type=int, default=1024)
    parser.add_argument("--rerank-provider", default="sentence_transformers")
    parser.add_argument("--rerank-model", default="D:/models/bge-reranker-v2-m3")
    parser.add_argument("--rerank-device", default="cpu")
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--rerank-max-length", type=int, default=512)
    parser.add_argument("--rerank-local-files-only", action="store_true")
    parser.add_argument("--recursion-limit", type=int, default=30)

    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    runtime = ToolAgentRuntimeConfig(
        project_root=args.project_root,
        chunks_path=args.chunks_path,
        index_path=args.index_path,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        embedding_api_key=args.embedding_api_key,
        embedding_timeout_seconds=args.embedding_timeout_seconds,
        mock_dimension=args.mock_dimension,
        rerank_provider=args.rerank_provider,
        rerank_model=args.rerank_model,
        rerank_device=args.rerank_device,
        rerank_batch_size=args.rerank_batch_size,
        rerank_max_length=args.rerank_max_length,
        rerank_local_files_only=args.rerank_local_files_only,
    )
    model_config = LangChainModelConfig.from_env()
    dependencies = build_tool_agent_dependencies(
        runtime=runtime,
        model_config=model_config,
    )
    service = CodeDocToolAgentService(
        runtime=runtime,
        dependencies=dependencies,
    )
    cases = load_agent_eval_cases(args.dataset)
    report = await run_agent_evaluation(
        cases=cases,
        executor=ToolAgentEvalExecutor(
            service=service,
            recursion_limit=args.recursion_limit,
        ),
        dataset_path=args.dataset,
        model_provider=model_config.provider,
        model_name=model_config.model_name,
    )
    result = save_agent_eval_report(
        report=report,
        output_path=args.output,
    )
    print(result)
    print(report.summary.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
