import re
from typing import Dict, List


SOURCE_PATTERN = re.compile(r"\[Source\s+(\d+)\]")


def evaluate_answer_quality(
    answer: str,
    citations: List[Dict],
) -> Dict:
    """
    检查回答中使用的 Source 引用是否真实存在。
    """
    if not answer or not answer.strip():
        raise ValueError("answer 不能为空")

    valid_citation_ids = {
        citation["citation_id"]
        for citation in citations
    }

    matched_numbers = SOURCE_PATTERN.findall(answer)

    used_citation_ids = {
        f"Source {number}"
        for number in matched_numbers
    }

    valid_used_ids = used_citation_ids & valid_citation_ids
    invalid_ids = used_citation_ids - valid_citation_ids

    warnings = []

    if citations and not used_citation_ids:
        warnings.append("回答没有使用任何引用标记")

    if invalid_ids:
        warnings.append("回答引用了不存在的来源")

    if not citations:
        warnings.append("当前没有可用的检索引用")

    used_count = len(used_citation_ids)

    if used_count == 0:
        valid_citation_rate = 0.0
    else:
        valid_citation_rate = len(valid_used_ids) / used_count

    return {
        "is_valid": len(invalid_ids) == 0,
        "has_citations": bool(valid_used_ids),
        "used_citation_ids": sorted(used_citation_ids),
        "valid_citation_ids": sorted(valid_used_ids),
        "invalid_citation_ids": sorted(invalid_ids),
        "valid_citation_rate": valid_citation_rate,
        "warnings": warnings,
    }
