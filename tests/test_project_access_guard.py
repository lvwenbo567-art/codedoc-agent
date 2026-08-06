import pytest

from security.project_access_guard import ProjectAccessGuard
from security.security_models import ProjectAccessDeniedError, RequestPrincipal


def test_project_access_is_isolated_by_allowed_project_ids() -> None:
    principal = RequestPrincipal(user_id="u1", allowed_project_ids=frozenset({1}))
    guard = ProjectAccessGuard()
    guard.require_access(principal=principal, project_id=1)
    with pytest.raises(ProjectAccessDeniedError):
        guard.require_access(principal=principal, project_id=2)
