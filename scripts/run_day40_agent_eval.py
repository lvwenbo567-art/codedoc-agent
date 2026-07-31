from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
sys.path.append(str(APP_DIR))

from evaluation.agent_eval_dataset import load_agent_eval_cases
from evaluation.agent_eval_report import save_agent_eval_report
from evaluation.agent_eval_runner import (
    ToolAgentEvalExecutor,
    run_agent_evaluation,
)
from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import (
    build_tool_agent_dependencies,
)
from langgraph_agent.tool_agent_service import CodeDocToolAgentService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Day40 CodeDoc Agent Evaluation."
    )
    parser.add_argument(
        "--dataset",
        default="data/evaluation/codedoc_agent_eval.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs/day40_agent_eval_report.json",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--chunks-path", default="outputs/test_project_chunks.json")
    parser.add_argument(
        "--index-path",
        default="outputs/test_project_vector_index_bge_m3.json",
    )
    parser.add_argument("--recursion-limit", type=int, default=30)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    runtime = ToolAgentRuntimeConfig(
        project_root=args.project_root,
        chunks_path=args.chunks_path,
        index_path=args.index_path,
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
