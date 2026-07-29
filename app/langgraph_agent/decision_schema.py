from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


QueryType = Literal[
    "code",
    "document",
    "structure",
    "unknown",
]

RetrievalStrategy = Literal[
    "original",
    "multi_query",
    "structure",
    "none",
]


class StrictDecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryDecision(StrictDecisionModel):
    query_type: QueryType
    retrieval_strategy: RetrievalStrategy
    symbol_name: str | None = Field(default=None, max_length=300)
    confidence: float = Field(ge=0, le=1)#置信度
    reason: str = Field(min_length=1, max_length=500)#判断理由
    decision_method: Literal[
        "model",
        "rule",
        "rule_fallback",
    ] = "model"


class EvidenceAssessment(StrictDecisionModel):
    sufficient: bool
    relevance_score: float = Field(ge=0, le=1)
    coverage_score: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=800)
    missing_information: list[str] = Field(default_factory=list)
    assessment_method: Literal[
        "model",
        "rule",
        "rule_fallback",
    ] = "model"
