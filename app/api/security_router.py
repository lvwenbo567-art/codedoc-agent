from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api.api_response import success_response
from security.path_guard import SafeProjectPathResolver
from security.prompt_injection_detector import PromptInjectionDetector
from security.security_models import PathAccessDeniedError
from security.sensitive_data_redactor import SensitiveDataRedactor


router = APIRouter(prefix="/security", tags=["security"])


class StrictSecurityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextSecurityRequest(StrictSecurityRequest):
    text: str = Field(min_length=1, max_length=50000)


class ProjectPathValidationRequest(StrictSecurityRequest):
    project_root: str = Field(min_length=1, max_length=1000)
    source_path: str = Field(min_length=1, max_length=1000)


@router.post("/scan-prompt-injection")
def scan_prompt_injection(request: TextSecurityRequest) -> dict:
    """开发调试接口：扫描文本中的常见 Prompt Injection 特征。"""
    result = PromptInjectionDetector().scan(request.text)
    return success_response(
        data={
            "suspicious": result.suspicious,
            "risk_score": result.risk_score,
            "findings": [finding.__dict__ for finding in result.findings],
        }
    )


@router.post("/redact")
def redact_sensitive_data(request: TextSecurityRequest) -> dict:
    """开发调试接口：展示敏感字段在进入模型前的脱敏效果。"""
    result = SensitiveDataRedactor().redact(request.text)
    return success_response(data=result.__dict__)


@router.post("/validate-project-path")
def validate_project_path(request: ProjectPathValidationRequest) -> dict:
    """开发调试接口：验证目标文件是否在项目安全范围内。"""
    try:
        path = SafeProjectPathResolver(project_root=request.project_root).resolve_file(
            request.source_path
        )
    except (ValueError, PathAccessDeniedError) as exc:
        raise HTTPException(status_code=403, detail="项目文件访问被拒绝") from exc
    return success_response(data={"source_path": str(path)})
