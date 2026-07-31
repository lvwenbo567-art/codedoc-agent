from __future__ import annotations

import math

from evaluation.agent_eval_schema import (
    AgentEvalCase,
    AgentEvalCaseResult,
    AgentEvalObservation,
    AgentEvalSummary,
)


def _safe_divide(
    numerator: float,
    denominator: float,
    *,
    empty_value: float,
) -> float:
    """
    防止除以 0 的小工具函数。
    """
    if denominator == 0:
        return empty_value

    return numerator / denominator


def _normalize_name_set(values: list[str]) -> set[str]:
    """
    把工具名列表转为去空白后的集合。
    """
    return {value.strip() for value in values if value.strip()}


def evaluate_agent_case(
    *,
    case: AgentEvalCase,
    observation: AgentEvalObservation,
) -> AgentEvalCaseResult:
    """
    对单个 Agent 运行结果计算评测指标。
    """
    expected_tools = _normalize_name_set(case.expected_tool_names)
    actual_tools = _normalize_name_set(observation.tool_names)
    forbidden_tools = _normalize_name_set(case.forbidden_tool_names)

    matched_tools = expected_tools & actual_tools
    missing_tools = sorted(expected_tools - actual_tools)
    unexpected_tools = sorted(actual_tools - expected_tools)
    forbidden_tools_used = sorted(actual_tools & forbidden_tools)

    if not expected_tools:
        tool_recall = 1.0#recall = 命中的预期工具数 / 预期工具总数 漏没漏
        tool_precision = 1.0 if not actual_tools else 0.0#precision = 命中的预期工具数 / 实际调用工具总数 准不准
    else:
        tool_recall = _safe_divide(
            len(matched_tools),
            len(expected_tools),
            empty_value=1.0,
        )
        tool_precision = _safe_divide(
            len(matched_tools),
            len(actual_tools),
            empty_value=0.0,
        )

    tool_f1 = _safe_divide(#F1 = 2 * P * R / (P + R)
        2 * tool_precision * tool_recall,
        tool_precision + tool_recall,
        empty_value=0.0,
    )
    tool_exact_match = float(expected_tools == actual_tools)

    if case.expected_first_tool is None:
        first_tool_accuracy = 1.0
    else:
        actual_first_tool = (
            observation.tool_names[0] if observation.tool_names else None
        )
        first_tool_accuracy = float(
            actual_first_tool == case.expected_first_tool
        )

    forbidden_tool_safety = float(not forbidden_tools_used)
    normalized_answer = observation.answer.lower()
    required_terms = [
        term.strip() for term in case.required_answer_terms if term.strip()
    ]
    missing_answer_terms = [
        term
        for term in required_terms
        if term.lower() not in normalized_answer
    ]
    answer_term_coverage = (
        1.0
        if not required_terms
        else _safe_divide(
            len(required_terms) - len(missing_answer_terms),
            len(required_terms),
            empty_value=1.0,
        )
    )
    completion_score = float(
        observation.status == case.expected_status
        and observation.stop_reason in case.accepted_stop_reasons
    )
    latency_pass = (
        1.0
        if case.max_latency_ms is None
        else float(observation.latency_ms <= case.max_latency_ms)
    )

    failure_reasons: list[str] = []

    if completion_score == 0:
        failure_reasons.append("执行状态或 stop_reason 不符合预期")

    if missing_tools:
        failure_reasons.append("缺少预期工具：" + ", ".join(missing_tools))

    if forbidden_tools_used:
        failure_reasons.append(
            "调用了禁止工具：" + ", ".join(forbidden_tools_used)
        )

    if first_tool_accuracy == 0:
        failure_reasons.append("第一次工具选择不符合预期")

    if missing_answer_terms:
        failure_reasons.append(
            "回答缺少关键词：" + ", ".join(missing_answer_terms)
        )

    if latency_pass == 0:
        failure_reasons.append("执行延迟超过阈值")

    if observation.error_message:
        failure_reasons.append("执行异常：" + observation.error_message)

    task_success = float(
        completion_score == 1
        and tool_recall == 1
        and forbidden_tool_safety == 1
        and first_tool_accuracy == 1
        and answer_term_coverage == 1
        and latency_pass == 1
    )

    return AgentEvalCaseResult(
        case_id=case.case_id,
        name=case.name,
        passed=bool(task_success),
        task_success=task_success,
        tool_exact_match=tool_exact_match,
        tool_precision=tool_precision,
        tool_recall=tool_recall,
        tool_f1=tool_f1,
        first_tool_accuracy=first_tool_accuracy,
        forbidden_tool_safety=forbidden_tool_safety,
        answer_term_coverage=answer_term_coverage,
        completion_score=completion_score,
        latency_pass=latency_pass,
        latency_ms=observation.latency_ms,
        expected_tools=sorted(expected_tools),
        actual_tools=observation.tool_names,
        missing_tools=missing_tools,
        unexpected_tools=unexpected_tools,
        forbidden_tools_used=forbidden_tools_used,
        missing_answer_terms=missing_answer_terms,
        failure_reasons=failure_reasons,
        observation=observation,
    )


def _average(values: list[float]) -> float:
    """
    计算平均值。
    """
    if not values:
        return 0.0

    return sum(values) / len(values)


def _percentile_95(values: list[float]) -> float:
    """
    计算简单 P95 延迟。
    """
    if not values:
        return 0.0

    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)

    return ordered[index]


def summarize_agent_eval_results(
    results: list[AgentEvalCaseResult],
) -> AgentEvalSummary:
    """
    汇总整批 Agent Evaluation 结果。
    """
    total_cases = len(results)
    passed_cases = sum(1 for result in results if result.passed)
    failed_cases = total_cases - passed_cases

    return AgentEvalSummary(
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        task_success_rate=_average([r.task_success for r in results]),
        tool_exact_match_rate=_average([r.tool_exact_match for r in results]),
        average_tool_precision=_average([r.tool_precision for r in results]),
        average_tool_recall=_average([r.tool_recall for r in results]),
        average_tool_f1=_average([r.tool_f1 for r in results]),
        first_tool_accuracy=_average([r.first_tool_accuracy for r in results]),
        forbidden_tool_safety_rate=_average(
            [r.forbidden_tool_safety for r in results]
        ),
        average_answer_term_coverage=_average(
            [r.answer_term_coverage for r in results]
        ),
        completion_rate=_average([r.completion_score for r in results]),
        latency_pass_rate=_average([r.latency_pass for r in results]),
        average_latency_ms=_average([r.latency_ms for r in results]),
        p95_latency_ms=_percentile_95([r.latency_ms for r in results]),
    )
