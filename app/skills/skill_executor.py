from __future__ import annotations

from skills.skill_schema import SkillDefinition


def build_skill_execution_plan(skill: SkillDefinition) -> dict:
    return {
        "skill_name": skill.skill_name,
        "display_name": skill.display_name,
        "recommended_tools": skill.recommended_tools,
        "requires_human_review_tools": skill.requires_human_review_tools,
        "steps": [
            {
                "step_name": step.step_name,
                "description": step.description,
                "recommended_tools": step.recommended_tools,
                "required": step.required,
            }
            for step in skill.steps
        ],
        "output_sections": skill.output_sections,
    }
