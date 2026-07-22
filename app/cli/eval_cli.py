import argparse
import json
from pathlib import Path

from logger import setup_logger
from evaluation.retrieval_eval import evaluate_from_json


def load_eval_queries(eval_path: str):
    """
    读取人工标注的评估 query 文件。
    """
    path = Path(eval_path)

    if not path.exists():
        raise FileNotFoundError(f"评估文件不存在：{eval_path}")

    content = path.read_text(encoding="utf-8")
    return json.loads(content)


def main() -> None:
    """
    命令行入口：读取评估集并输出检索指标。
    """
    logger = setup_logger()
    logger.info("开始运行检索评估 CLI")

    parser = argparse.ArgumentParser(description="评估 chunks 检索效果")

    parser.add_argument(
        "--chunks_path",
        default="outputs/chunks.json",
        help="chunks JSON 文件路径",
    )

    parser.add_argument(
        "--eval_path",
        required=True,
        help="评估 query JSON 文件路径",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="检索返回的 Top-K 数量",
    )

    args = parser.parse_args()

    eval_queries = load_eval_queries(args.eval_path)

    result = evaluate_from_json(
        chunks_path=args.chunks_path,
        eval_queries=eval_queries,
        top_k=args.top_k,
    )

    summary = result["summary"]

    print("检索评估结果:")
    print(f"- chunks_path: {result['chunks_path']}")
    print(f"- top_k: {result['top_k']}")
    print(f"- query_count: {summary['query_count']}")
    print(f"- avg_hit_rate: {summary['avg_hit_rate']:.4f}")
    print(f"- avg_recall: {summary['avg_recall']:.4f}")
    print(f"- avg_mrr: {summary['avg_mrr']:.4f}")
    print()

    print("单条 query 结果:")
    for item in summary["items"]:
        print(f"- query: {item['query']}")
        print(f"  hit_rate: {item['hit_rate']:.4f}")
        print(f"  recall: {item['recall']:.4f}")
        print(f"  mrr: {item['mrr']:.4f}")
        print(f"  retrieved_chunk_ids: {item['retrieved_chunk_ids']}")
        print(f"  expected_chunk_ids: {item['expected_chunk_ids']}")
        print()

    logger.info(
        "检索评估 CLI 运行结束，query_count=%s, avg_hit_rate=%.4f, avg_recall=%.4f, avg_mrr=%.4f",
        summary["query_count"],
        summary["avg_hit_rate"],
        summary["avg_recall"],
        summary["avg_mrr"],
    )


if __name__ == "__main__":
    main()
