from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictSkillModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SkillStep(StrictSkillModel):
    step_name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    recommended_tools: list[str] = Field(default_factory=list)
    required: bool = True


class SkillDefinition(StrictSkillModel):
    skill_name: str = Field(min_length=1, max_length=80)
    display_name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=1000)
    intent_keywords: list[str] = Field(default_factory=list)
    example_queries: list[str] = Field(default_factory=list)
    recommended_tools: list[str] = Field(default_factory=list)
    output_sections: list[str] = Field(default_factory=list)
    steps: list[SkillStep] = Field(default_factory=list)
    requires_human_review_tools: list[str] = Field(default_factory=list)


class SkillRouteRequest(StrictSkillModel):
    query: str = Field(min_length=1, max_length=1000)


class SkillRouteResult(StrictSkillModel):
    query: str
    skill_name: str
    display_name: str
    confidence: float = Field(ge=0, le=1)
    matched_keywords: list[str] = Field(default_factory=list)
    recommended_tools: list[str] = Field(default_factory=list)
    output_sections: list[str] = Field(default_factory=list)
    reason: str = ""
