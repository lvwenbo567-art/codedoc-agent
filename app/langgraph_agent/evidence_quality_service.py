from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.model_factory import create_chat_model
from langgraph_agent.decision_schema import EvidenceAssessment
from langgraph_agent.state import GraphEvidence, QueryType


EVIDENCE_JUDGE_SYSTEM_PROMPT = """
你是 CodeDoc Evidence Judge。

请判断检索证据是否足以支持回答用户问题。
只根据给定证据判断，不要补充不存在的信息。
如果证据与问题无关、内容太短或只有文件名没有内容，应判断为不足。
""".strip()


class EvidenceQualityService:
    def __init__(
        self,
        *,
        model_config: LangChainModelConfig,
        max_evidence_items: int = 4,
        max_chars_per_item: int = 1200,
    ) -> None:
        self.model_config = model_config
        self.max_evidence_items = max_evidence_items
        self.max_chars_per_item = max_chars_per_item
        self._structured_model = None

        if model_config.provider != "mock":
            model = create_chat_model(model_config)
            self._structured_model = model.with_structured_output(EvidenceAssessment)

    @staticmethod
    def _has_exact_symbol_match(
        evidence: list[GraphEvidence],
        symbol_name: str | None,
    ) -> bool:
        '''
        证据里有没有命中用户指定的代码符号。
        '''
        if not symbol_name:
            return False

        candidates = {symbol_name, symbol_name.split(".")[-1]}
        '''
        构造候选名。
如果：
symbol_name = "RerankClient.score"
那么：
candidates = {
    "RerankClient.score",
    "score",
}
        '''
        for item in evidence:
            values = {
                str(item.get("symbol_name") or ""),
                str(item.get("qualified_name") or ""),
                str(item.get("chunk_id") or ""),
                str(item.get("content") or ""),
            }

            if any(candidate in value for candidate in candidates for value in values):
                return True

        return False

    def _rule_assessment(
        self,
        *,
        query_type: QueryType,
        evidence: list[GraphEvidence],
        symbol_name: str | None,
        method: str,
    ) -> EvidenceAssessment:
        valid_items = [
            item
            for item in evidence
            if str(item.get("content") or "").strip()
        ]#过滤出有内容的证据。如果 content 为空，就不算有效证据。

        if not valid_items:
            return EvidenceAssessment(
                sufficient=False,
                relevance_score=0,
                coverage_score=0,
                reason="没有检索到包含有效内容的证据",
                missing_information=["缺少项目证据"],
                assessment_method=method,
            )

        if query_type == "structure":
            return EvidenceAssessment(
                sufficient=True,
                relevance_score=0.9,
                coverage_score=0.8,
                reason="已经获得项目目录或模块结构证据",
                missing_information=[],
                assessment_method=method,
            )

        if self._has_exact_symbol_match(valid_items, symbol_name):
            return EvidenceAssessment(
                sufficient=True,
                relevance_score=0.95,
                coverage_score=0.85,
                reason="证据中包含用户指定的精确代码符号",
                missing_information=[],
                assessment_method=method,
            )
 
        total_content_chars = sum(
            len(str(item.get("content") or ""))
            for item in valid_items
        )

        if total_content_chars < 80:
            return EvidenceAssessment(
                sufficient=False,
                relevance_score=0.25,
                coverage_score=0.2,
                reason="证据内容过短，不足以可靠回答",
                missing_information=["缺少足够上下文"],
                assessment_method=method,
            )

        return EvidenceAssessment(
            sufficient=True,
            relevance_score=0.75,
            coverage_score=0.65,
            reason="证据包含可用于回答问题的项目内容",
            missing_information=[],
            assessment_method=method,
        )

    def _build_prompt(
        self,
        *,
        query: str,
        query_type: QueryType,
        evidence: list[GraphEvidence],
        symbol_name: str | None,
    ) -> str:
        compact_items = []

        for index, item in enumerate(
            evidence[: self.max_evidence_items],
            start=1,
        ):
            content = str(item.get("content") or "")
            compact_items.append(
                {
                    "source": index,
                    "source_path": item.get("source_path"),
                    "chunk_id": item.get("chunk_id"),
                    "symbol_name": item.get("symbol_name"),
                    "qualified_name": item.get("qualified_name"),
                    "content": content[: self.max_chars_per_item],
                }
            )

        return json.dumps(
            {
                "query": query,
                "query_type": query_type,
                "symbol_name": symbol_name,
                "evidence": compact_items,
            },
            ensure_ascii=False,
            indent=2,
        )

    def assess(
        self,
        *,
        query: str,
        query_type: QueryType,
        evidence: list[GraphEvidence],
        symbol_name: str | None = None,
    ) -> EvidenceAssessment:
        rule_assessment = self._rule_assessment(
            query_type=query_type,
            evidence=evidence,
            symbol_name=symbol_name,
            method="rule",
        )

        if self._structured_model is None:
            return rule_assessment

        try:
            assessment = self._structured_model.invoke(
                [
                    SystemMessage(content=EVIDENCE_JUDGE_SYSTEM_PROMPT),
                    HumanMessage(
                        content=self._build_prompt(
                            query=query,
                            query_type=query_type,
                            evidence=evidence,
                            symbol_name=symbol_name,
                        )
                    ),
                ]
            )

            if not isinstance(assessment, EvidenceAssessment):
                assessment = EvidenceAssessment.model_validate(assessment)

            return assessment.model_copy(update={"assessment_method": "model"})

        except Exception:
            return self._rule_assessment(
                query_type=query_type,
                evidence=evidence,
                symbol_name=symbol_name,
                method="rule_fallback",
            )
