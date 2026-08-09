from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.api_response import success_response
from skills import build_builtin_skill_registry
from skills.skill_executor import build_skill_execution_plan
from skills.skill_schema import SkillRouteRequest


router = APIRouter(
    prefix="/skills",
    tags=["skills"],
)


def _registry():
    return build_builtin_skill_registry()


@router.get("")
def list_skills() -> dict:
    registry = _registry()
    return success_response(
        data={
            "skills": [
                skill.model_dump()
                for skill in registry.list_skills()
            ]
        }
    )


@router.get("/{skill_name}")
def get_skill(skill_name: str) -> dict:
    skill = _registry().get(skill_name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")

    return success_response(
        data={
            "skill": skill.model_dump(),
            "plan": build_skill_execution_plan(skill),
        }
    )


@router.post("/route")
def route_skill(request: SkillRouteRequest) -> dict:
    result = _registry().route(request.query)
    skill = _registry().require(result.skill_name)
    return success_response(
        data={
            "route": result.model_dump(),
            "plan": build_skill_execution_plan(skill),
        }
    )
