import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from config import VECTOR_INDEX_FORMAT_VERSION


def cosine_similarity(
    vector_a: List[float],
    vector_b: List[float],
) -> float:
    """
    计算两个向量的余弦相似度，分数越高表示越相关。
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


def build_index_metadata(
    embedding_provider: str,
    embedding_model: str,
    dimension: int,
    normalized: bool,
    record_count: int,
    build_stats: Dict | None = None,
) -> Dict:
    """
    构建向量索引元数据，用于记录模型、维度、版本和建库统计。
    """
    metadata = {
        "index_format_version": VECTOR_INDEX_FORMAT_VERSION,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "dimension": dimension,
        "normalized": normalized,
        "record_count": record_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if build_stats is not None:
        metadata["build_stats"] = build_stats

    return metadata


def save_vector_index(
    records: List[Dict],
    output_path: str,
    metadata: Dict | None = None,
) -> Path:
    """
    通过临时文件原子性保存向量索引，避免写到一半破坏旧索引。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    if metadata is None:
        # Compatibility for tests and old callers.
        temp_path.write_text(
            json.dumps(
                records,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temp_path.replace(path)
        return path

    payload = {
        "metadata": metadata,
        "records": records,
    }

    temp_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temp_path.replace(path)

    return path


def load_vector_index_bundle(
    input_path: str,
) -> Dict:
    """
    读取完整向量索引，包括 metadata 和 records。
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"向量索引文件不存在：{input_path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    # Compatibility with Day19/20 list-only indexes.
    if isinstance(data, list):
        return {
            "metadata": {
                "index_format_version": "legacy",
            },
            "records": data,
        }

    if not isinstance(data, dict):
        raise ValueError("向量索引格式不正确")

    metadata = data.get("metadata")
    records = data.get("records")

    if not isinstance(metadata, dict):
        raise ValueError("向量索引缺少 metadata")

    if not isinstance(records, list):
        raise ValueError("向量索引缺少 records")

    return {
        "metadata": metadata,
        "records": records,
    }


def load_vector_index(input_path: str) -> List[Dict]:
    """
    兼容旧调用方式，只返回向量记录列表。
    """
    bundle = load_vector_index_bundle(input_path)
    return bundle["records"]
