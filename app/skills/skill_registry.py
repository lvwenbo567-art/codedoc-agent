from __future__ import annotations

from skills.skill_schema import SkillDefinition, SkillRouteResult


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        if skill.skill_name in self._skills:
            raise ValueError(f"Skill already registered: {skill.skill_name}")
        self._skills[skill.skill_name] = skill

    def get(self, skill_name: str) -> SkillDefinition | None:
        return self._skills.get(skill_name)

    def require(self, skill_name: str) -> SkillDefinition:
        skill = self.get(skill_name)
        if skill is None:
            raise KeyError(f"Skill not found: {skill_name}")
        return skill

    def list_skills(self) -> list[SkillDefinition]:
        return [self._skills[name] for name in sorted(self._skills)]

    def route(self, query: str) -> SkillRouteResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        lower_query = normalized_query.lower()
        best_skill: SkillDefinition | None = None
        best_matches: list[str] = []

        for skill in self.list_skills():
            matches = [
                keyword
                for keyword in skill.intent_keywords
                if keyword.lower() in lower_query
            ]
            if len(matches) > len(best_matches):
                best_skill = skill
                best_matches = matches

        if best_skill is None:
            best_skill = self.require("project_onboarding")

        confidence = min(1.0, len(best_matches) / 3)
        if not best_matches:
            confidence = 0.2

        return SkillRouteResult(
            query=normalized_query,
            skill_name=best_skill.skill_name,
            display_name=best_skill.display_name,
            confidence=round(confidence, 3),
            matched_keywords=best_matches,
            recommended_tools=best_skill.recommended_tools,
            output_sections=best_skill.output_sections,
            reason=(
                "Matched intent keywords."
                if best_matches
                else "No strong keyword match; fallback to project onboarding."
            ),
        )
