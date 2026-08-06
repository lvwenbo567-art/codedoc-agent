from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RequestPrincipal:
    """经认证层解析后的请求身份；Day43 不实现登录，只定义授权边界。"""

    user_id: str
    allowed_project_ids: frozenset[int]
    '''
    不需要顺序；
    不应有重复值；
    主要操作是判断“某个项目是否在集合中”
    权限信息不应在一次请求处理中被随意增加或删除
    '''
    roles: frozenset[str] = field(default_factory=lambda: frozenset({"project_reader"}))


class ProjectAccessDeniedError(PermissionError):
    """用户无权访问目标项目。"""


class PathAccessDeniedError(PermissionError):
    """请求路径超出项目安全范围或命中敏感文件策略。"""

