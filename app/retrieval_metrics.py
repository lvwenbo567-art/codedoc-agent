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
