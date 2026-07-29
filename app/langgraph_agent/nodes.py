from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from langgraph_agent.decision_schema import EvidenceAssessment, QueryDecision
from langgraph_agent.dependencies import CodeDocGraphDependencies
from langgraph_agent.evidence_adapter import (
    compress_project_structure_evidence,
    convert_result_to_evidence,
)
from langgraph_agent.query_classifier import RuleBasedQueryClassifier
from langgraph_agent.state import CodeDocGraphState, GraphCitation, GraphEvidence


def _to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if is_dataclass(value):
        dumped = asdict(value)

        if isinstance(dumped, dict):
            return dumped

    if hasattr(value, "model_dump"):
        dumped = value.model_dump()

        if isinstance(dumped, dict):
            return dumped

    return {"content": str(value)}


def _extract_answer_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()

    mapping = _to_mapping(result)

    for key in ("answer", "content", "text"):
        value = mapping.get(key)

        if isinstance(value, str):
            return value.strip()

    return str(result).strip()


def _tool_result_success(result: Any) -> bool:
    return bool(_to_mapping(result).get("success"))


def _tool_result_data(result: Any) -> Any:
    return _to_mapping(result).get("data")


def _tool_result_error(result: Any) -> str | None:
    mapping = _to_mapping(result)
    return (
        mapping.get("error_message")
        or mapping.get("error_code")
    )


@dataclass
class CodeDocWorkflowNodes:
    """
    Day35/Day36 LangGraph 节点集合。

    Day35：规则分类的确定性 Workflow。
    Day36：模型/规则决策 + 真实 RAG Pipeline 的 Agentic RAG Workflow v1。
    """

    dependencies: CodeDocGraphDependencies
    classifier: RuleBasedQueryClassifier = field(
        default_factory=RuleBasedQueryClassifier,
    )

    def initialize_node(
        self,
        state: CodeDocGraphState,
    ) -> dict:
        query = str(state.get("query") or "").strip()

        return {
            "query": query,
            "query_type": "unknown",
            "retrieval_strategy": "none",
            "symbol_name": None,
            "answer": "",
            "citations": [],
            "evidence_sufficient": False,
            "error_message": None,
            "degraded": False,
            "degrade_reasons": [],
            "retrieval_metadata": {},
            "query_decision": {},
            "evidence_assessment": {},
            "answer_quality": {},
            "execution_steps": ["initialize"],
        }

    def classify_query_node(
        self,
        state: CodeDocGraphState,
    ) -> dict:
        query = str(state.get("query") or "")
        query_type = self.classifier.classify(query)

        return {
            "query_type": query_type,
            "execution_steps": ["classify_query"],
        }

    def analyze_query_node(
        self,
        state: CodeDocGraphState,
    ) -> dict:
        query = str(state.get("query") or "")
        service = self.dependencies.query_decision_service

        if service is None:
            query_type = self.classifier.classify(query)
            decision = QueryDecision(
                query_type=query_type,
                retrieval_strategy=(
                    "structure"
                    if query_type == "structure"
                    else "multi_query"
                    if query_type in {"code", "document"}
                    else "none"
                ),
                symbol_name=None,
                confidence=0.5,
                reason="未配置 QueryDecisionService，使用规则兜底",
                decision_method="rule_fallback",
            )
        else:
            decision = service.analyze(query)

        return {
            "query_type": decision.query_type,
            "retrieval_strategy": decision.retrieval_strategy,
            "symbol_name": decision.symbol_name,
            "query_decision": decision.model_dump(),
            "execution_steps": ["analyze_query"],
        }

    def _execute_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        evidence_type: str,
    ) -> tuple[list[GraphEvidence], str | None, Any]:
        try:
            result = self.dependencies.tool_executor.execute(
                tool_name=tool_name,
                arguments=arguments,
            )
        except Exception as exc:
            return [], f"工具执行异常：{tool_name}：{exc}", None

        if not _tool_result_success(result):
            return [], _tool_result_error(result) or f"工具执行失败：{tool_name}", result

        evidence = convert_result_to_evidence(
            data=_tool_result_data(result),
            evidence_type=evidence_type,
        )

        return evidence, None, result

    def _execute_search_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        evidence_type: str,
    ) -> tuple[list[GraphEvidence], str | None]:
        evidence, error_message, _ = self._execute_tool(
            tool_name=tool_name,
            arguments=arguments,
            evidence_type=evidence_type,
        )
        return evidence, error_message

    def exact_symbol_lookup_node(
        self,
        state: CodeDocGraphState,
    ) -> dict:
        symbol_name = str(state.get("symbol_name") or "").strip()

        if not symbol_name:
            return {
                "error_message": "没有可用于精确查询的 symbol_name",
                "execution_steps": ["exact_symbol_lookup"],
            }

        evidence, error_message, result = self._execute_tool(
            tool_name="get_symbol_definition",
            arguments={
                "symbol_name": symbol_name,
                "exact_match": True,
                "max_results": 5,
            },
            evidence_type="symbol_lookup",
        )
        result_mapping = _to_mapping(_tool_result_data(result) if result else {})
        result_count = int(result_mapping.get("result_count") or len(evidence))

        return {
            "evidence": evidence,
            "error_message": error_message,
            "retrieval_metadata": {
                "symbol_lookup_result_count": result_count,
                "symbol_lookup_hit": result_count > 0,
                "symbol_name": symbol_name,
            },
            "execution_steps": ["exact_symbol_lookup"],
        }

    def _search_arguments(
        self,
        *,
        query: str,
        query_strategy: str,
    ) -> dict[str, Any]:
        runtime = self.dependencies.runtime
        return {
            "query": query,
            "top_k": runtime.final_top_k if runtime else 5,
            "candidate_top_k": runtime.candidate_top_k if runtime else 20,
            "query_strategy": query_strategy,
        }

    def code_search_node(
        self,
        state: CodeDocGraphState,
    ) -> dict:#老
        evidence, error_message = self._execute_search_tool(
            tool_name="search_code",
            arguments={
                "query": state.get("query", ""),
                "top_k": 5,
                "candidate_top_k": 20,
                "query_strategy": "multi_query",
            },
            evidence_type="code_search",
        )

        return {
            "evidence": evidence,
            "error_message": error_message,
            "execution_steps": ["code_search"],
        }

    def code_retrieve_node(
        self,
        state: CodeDocGraphState,
    ) -> dict:
        query_strategy = str(state.get("retrieval_strategy") or "multi_query")

        if query_strategy == "structure":
            query_strategy = "multi_query"

        if query_strategy == "none":
            query_strategy = "original"

        evidence, error_message, result = self._execute_tool(
            tool_name="search_code",
            arguments=self._search_arguments(
                query=str(state.get("query") or ""),
                query_strategy=query_strategy,
            ),
            evidence_type="code_search",
        )
        data = _to_mapping(_tool_result_data(result) if result else {})
        degrade_reason = data.get("degrade_reason")

        return {
            "evidence": evidence,
            "error_message": error_message,
            "retrieval_metadata": {
                "query_strategy": data.get("query_strategy", query_strategy),
                "query_items": data.get("query_items"),
                "rewrite_result": data.get("rewrite_result"),
                "rerank_applied": data.get("rerank_applied"),
                "rerank_duration_ms": data.get("rerank_duration_ms"),
                "retrieval_mode": data.get("retrieval_mode"),
                "candidate_count": data.get("candidate_count"),
                "result_count": data.get("result_count", len(evidence)),
            },
            "degraded": bool(data.get("degraded")),
            "degrade_reasons": [str(degrade_reason)] if degrade_reason else [],
            "execution_steps": ["code_retrieve"],
        }

    def document_search_node(
        self,
        state: CodeDocGraphState,
    ) -> dict:#老
        evidence, error_message = self._execute_search_tool(
            tool_name="search_documents",
            arguments={
                "query": state.get("query", ""),
                "top_k": 5,
                "candidate_top_k": 20,
                "query_strategy": "multi_query",
            },
            evidence_type="document_search",
        )

        return {
            "evidence": evidence,
            "error_message": error_message,
            "execution_steps": ["document_search"],
        }

    def document_retrieve_node(
        self,
        state: CodeDocGraphState,
    ) -> dict:
        query_strategy = str(state.get("retrieval_strategy") or "multi_query")

        if query_strategy not in {"original", "multi_query"}:
            query_strategy = "multi_query"

        evidence, error_message, result = self._execute_tool(
            tool_name="search_documents",
            arguments=self._search_arguments(
                query=str(state.get("query") or ""),
                query_strategy=query_strategy,
            ),
            evidence_type="document_search",
        )
        data = _to_mapping(_tool_result_data(result) if result else {})
        degrade_reason = data.get("degrade_reason")

        return {
            "evidence": evidence,
            "error_message": error_message,
            "retrieval_metadata": {
                "query_strategy": data.get("query_strategy", query_strategy),
                "query_items": data.get("query_items"),
                "rewrite_result": data.get("rewrite_result"),
                "rerank_applied": data.get("rerank_applied"),
                "rerank_duration_ms": data.get("rerank_duration_ms"),
                "retrieval_mode": data.get("retrieval_mode"),
                "candidate_count": data.get("candidate_count"),
                "result_count": data.get("result_count", len(evidence)),
            },
            "degraded": bool(data.get("degraded")),
            "degrade_reasons": [str(degrade_reason)] if degrade_reason else [],
            "execution_steps": ["document_retrieve"],
        }

    def project_structure_node(
        self,
        state: CodeDocGraphState,
    ) -> dict:
        evidence, error_message = self._execute_search_tool(
            tool_name="get_project_structure",
            arguments={
                "max_depth": 4,
                "max_entries": 300,
                "include_files": True,
                "include_hidden": False,
            },
            evidence_type="project_structure",
        )
        evidence = compress_project_structure_evidence(evidence)

        return {
            "evidence": evidence,
            "error_message": error_message,
            "execution_steps": ["project_structure"],
        }

    def check_evidence_node(
        self,
        state: CodeDocGraphState,
    ) -> dict:#老
        evidence = state.get("evidence") or []
        evidence_sufficient = bool(evidence)

        return {
            "evidence_sufficient": evidence_sufficient,
            "execution_steps": ["check_evidence"],
        }

    def assess_evidence_node(
        self,
        state: CodeDocGraphState,
    ) -> dict:
        service = self.dependencies.evidence_quality_service
        evidence = list(state.get("evidence") or [])

        if service is None:
            assessment = EvidenceAssessment(
                sufficient=bool(evidence),
                relevance_score=0.5 if evidence else 0,
                coverage_score=0.5 if evidence else 0,
                reason="未配置 EvidenceQualityService，使用规则兜底",
                missing_information=[] if evidence else ["缺少项目证据"],
                assessment_method="rule_fallback",
            )
        else:
            assessment = service.assess(
                query=str(state.get("query") or ""),
                query_type=state.get("query_type", "unknown"),
                evidence=evidence,
                symbol_name=state.get("symbol_name"),
            )

        return {
            "evidence_sufficient": assessment.sufficient,
            "evidence_assessment": assessment.model_dump(),
            "execution_steps": ["assess_evidence"],
        }

    def _build_citations(
        self,
        evidence: list[GraphEvidence],
    ) -> list[GraphCitation]:
        citations: list[GraphCitation] = []

        for index, item in enumerate(evidence, start=1):
            citations.append(
                GraphCitation(
                    citation_id=f"Source {index}",
                    source_path=item.get("source_path", ""),
                    chunk_id=item.get("chunk_id"),
                    score=item.get("score"),
                    start_line=item.get("start_line"),
                    end_line=item.get("end_line"),
                )
            )

        return citations

    def _build_answer_prompt(
        self,
        *,
        query: str,
        evidence: list[GraphEvidence],
    ) -> str:
        context_parts = []

        for index, item in enumerate(evidence, start=1):
            context_parts.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"source_path: {item.get('source_path', '')}",
                        f"content: {item.get('content', '')}",
                    ]
                )
            )

        context = "\n\n".join(context_parts)

        return (
            "请严格基于下面的项目证据回答用户问题。"
            "如果证据不足，请明确说明。"
            "解释代码时不能补充证据中不存在的逻辑。"
            "回答中尽量使用 [Source N] 引用。\n\n"
            f"用户问题：{query}\n\n"
            f"项目证据：\n{context}"
        )

    def build_answer_node(
        self,
        state: CodeDocGraphState,
    ) -> dict:
        query = str(state.get("query") or "")
        evidence = list(state.get("evidence") or [])

        if self.dependencies.answer_service is not None:
            try:
                if hasattr(
                    self.dependencies.answer_service,
                    "generate",
                ):
                    runtime = self.dependencies.runtime
                    result = self.dependencies.answer_service.generate(
                        query=query,
                        evidence=evidence,
                        max_context_chars=(
                            runtime.max_context_chars
                            if runtime
                            else 6000
                        ),
                    )
                    result_mapping = _to_mapping(result)
                else:
                    result_mapping = (
                        self.dependencies.answer_service.build_answer(
                            query=query,
                            evidence=evidence,
                        )
                    )

                return {
                    "answer": result_mapping["answer"],
                    "citations": result_mapping["citations"],
                    "answer_quality": result_mapping["answer_quality"],
                    "execution_steps": ["build_answer"],
                }
            except Exception as exc:
                return {
                    "answer": f"已找到项目证据，但生成回答失败：{exc}",
                    "citations": self._build_citations(evidence),
                    "error_message": str(exc),
                    "execution_steps": ["build_answer"],
                }

        citations = self._build_citations(evidence)
        prompt = self._build_answer_prompt(query=query, evidence=evidence)

        try:
            result = self.dependencies.chat_service.ask(query=prompt, history=[])
            answer = _extract_answer_text(result)
        except Exception as exc:
            answer = f"已找到项目证据，但生成回答失败：{exc}"

        if not answer:
            answer = "已找到项目证据，但模型没有返回有效回答。"

        return {
            "answer": answer,
            "citations": citations,
            "execution_steps": ["build_answer"],
        }

    def fallback_answer_node(
        self,
        state: CodeDocGraphState,
    ) -> dict:
        query_type = state.get("query_type", "unknown")
        error_message = state.get("error_message")
        assessment = state.get("evidence_assessment") or {}

        if query_type == "unknown":
            answer = (
                "无法确定该问题属于代码、文档还是项目结构问题，"
                "因此本次工作流没有执行检索。"
            )
        elif assessment:
            answer = (
                "没有获得足够项目证据，无法可靠回答。"
                f"原因：{assessment.get('reason')}"
            )
        elif error_message:
            answer = f"没有获得足够项目证据，无法可靠回答。原因：{error_message}"
        else:
            answer = "没有获得足够项目证据，无法可靠回答。"

        return {
            "answer": answer,
            "evidence_sufficient": False,
            "execution_steps": ["fallback_answer"],
        }
