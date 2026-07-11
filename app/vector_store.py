import json
import math
from pathlib import Path
from typing import Dict, List


def cosine_similarity(
    vector_a: List[float],
    vector_b: List[float],
) -> float:
    """
    计算两个向量的余弦相似度。
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("两个向量的维度必须一致")

    if not vector_a:
        raise ValueError("向量不能为空")

    dot_product = sum(
        value_a * value_b
        for value_a, value_b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def save_vector_index(
    records: List[Dict],
    output_path: str,
) -> Path:
    """
    将向量索引保存为 JSON 文件。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def load_vector_index(input_path: str) -> List[Dict]:
    """
    从 JSON 文件读取向量索引。
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"向量索引文件不存在：{input_path}")

    content = path.read_text(encoding="utf-8")

    return json.loads(content)