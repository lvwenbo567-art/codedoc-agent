from __future__ import annotations

from security.security_models import ProjectAccessDeniedError, RequestPrincipal


class ProjectAccessGuard:
    """在检索、工具和持久化操作之前校验项目归属。"""

    def require_access(self, *, principal: RequestPrincipal, project_id: int) -> None:
        if project_id <= 0:
            raise ValueError("project_id 必须大于 0")
        if not principal.user_id.strip():
            raise ProjectAccessDeniedError("未认证用户不能访问项目")
        if project_id not in principal.allowed_project_ids:
            raise ProjectAccessDeniedError("当前用户无权访问该项目")

