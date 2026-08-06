def extract_chunk_ids(results: list[dict]) -> list[str]:
    """
    从检索结果中提取 chunk_id 列表。
    """
    return [
        result["chunk_id"]
        for result in results
    ]


def calculate_hit_at_k(
    results: list[dict],
    expected_chunk_ids: list[str],
    k: int,
) -> float:
    """
    计算 Hit@K：Top-K 中只要命中任意一个正确 chunk，就记为 1。
    """
    if k <= 0:
        raise ValueError("k 必须大于 0")

    expected = set(expected_chunk_ids)
    retrieved = set(extract_chunk_ids(results)[:k])

    return 1.0 if expected & retrieved else 0.0


def calculate_reciprocal_rank(
    results: list[dict],
    expected_chunk_ids: list[str],
) -> float:
    """
    计算 Reciprocal Rank：第一个正确结果排名的倒数。
    """
    expected = set(expected_chunk_ids)

    for rank, result in enumerate(results, start=1):
        if result["chunk_id"] in expected:
            return 1.0 / rank

    return 0.0


def calculate_recall_at_k(
    results: list[dict],
    expected_chunk_ids: list[str],
    k: int,
) -> float:
    """
    计算 Recall@K：Top-K 命中的正确 chunk 数 / 正确 chunk 总数。
    """
    if k <= 0:
        raise ValueError("k 必须大于 0")

    if not expected_chunk_ids:
        return 0.0

    expected = set(expected_chunk_ids)
    retrieved = set(extract_chunk_ids(results)[:k])

    return len(expected & retrieved) / len(expected)


def calculate_ndcg_at_k(
    results: list[dict],
    expected_chunk_ids: list[str],
    k: int,
) -> float:
    """
    计算二值相关性的 NDCG@K。
    """
    if k <= 0:
        raise ValueError("k 必须大于 0")

    if not expected_chunk_ids:
        return 0.0

    expected = set(expected_chunk_ids)
    gains: list[float] = []

    for index, result in enumerate(results[:k], start=1):
        relevance = 1.0 if result["chunk_id"] in expected else 0.0
        gains.append(relevance / _log2(index + 1))

    dcg = sum(gains)
    ideal_hits = min(len(expected), k)
    ideal_dcg = sum(
        1.0 / _log2(index + 1)
        for index in range(1, ideal_hits + 1)
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def _log2(value: int) -> float:
    import math

    return math.log2(value)
