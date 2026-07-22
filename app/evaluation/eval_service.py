import json
from pathlib import Path
from typing import Dict, List

from evaluation.retrieval_eval import evaluate_from_json


def load_eval_queries(eval_path: str) -> List[Dict]:
    """
    读取人工标注的评估 query 文件。
    """
    path = Path(eval_path)

    if not path.exists():
        raise FileNotFoundError(f"评估文件不存在：{eval_path}")

    content = path.read_text(encoding="utf-8")

    return json.loads(content)


def evaluate_retrieval_from_files(
    chunks_path: str,
    eval_path: str,
    top_k: int = 5,
) -> Dict:
    """
    从 chunks 文件和 eval 文件中执行检索评估。
    """
    eval_queries = load_eval_queries(eval_path)

    result = evaluate_from_json(
        chunks_path=chunks_path,
        eval_queries=eval_queries,
        top_k=top_k,
    )

    return result