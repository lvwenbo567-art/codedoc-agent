from typing import Any, Dict, Optional

from fastapi.responses import JSONResponse


class UTF8JSONResponse(JSONResponse):
    """
    统一使用 UTF-8 字符集返回 JSON。

    FastAPI/Starlette 默认的 JSON 响应头通常只有 application/json。
    大多数客户端会按 JSON 规范将其当作 UTF-8，但 Windows PowerShell
    在部分版本中会按本地代码页解码未声明 charset 的响应，导致中文乱码。
    """

    media_type = "application/json; charset=utf-8"


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
    if status_code == 502:
        return "MODEL_SERVICE_ERROR"

    if status_code == 504:
        return "MODEL_SERVICE_TIMEOUT"
    return "HTTP_ERROR"
