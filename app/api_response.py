from typing import Any, Dict, Optional


def success_response(data: Any = None) -> Dict[str, Any]:
    """
    构造统一成功响应。
    """
    return {
        "success": True,
        "data": data,
    }


def error_response(
    code: str,
    message: str,
    details: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    构造统一失败响应。
    """
    error = {
        "code": code,
        "message": message,
    }

    if details is not None:
        error["details"] = details

    return {
        "success": False,
        "error": error,
    }


def http_status_to_error_code(status_code: int) -> str:
    """
    将 HTTP 状态码转换为统一错误码。
    """
    if status_code == 400:
        return "BAD_REQUEST"

    if status_code == 404:
        return "NOT_FOUND"

    if status_code == 422:
        return "VALIDATION_ERROR"

    if status_code == 500:
        return "INTERNAL_SERVER_ERROR"

    return "HTTP_ERROR"