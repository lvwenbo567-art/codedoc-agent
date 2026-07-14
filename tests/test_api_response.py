from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from api_response import (
    error_response,
    http_status_to_error_code,
    success_response,
)


def test_success_response():
    response = success_response(
        data={
            "message": "ok",
        }
    )

    assert response["success"] is True
    assert response["data"]["message"] == "ok"


def test_error_response_without_details():
    response = error_response(
        code="BAD_REQUEST",
        message="参数错误",
    )

    assert response["success"] is False
    assert response["error"]["code"] == "BAD_REQUEST"
    assert response["error"]["message"] == "参数错误"
    assert "details" not in response["error"]


def test_error_response_with_details():
    response = error_response(
        code="VALIDATION_ERROR",
        message="请求参数校验失败",
        details=[{"field": "query"}],
    )

    assert response["success"] is False
    assert response["error"]["code"] == "VALIDATION_ERROR"
    assert response["error"]["details"] == [{"field": "query"}]


def test_http_status_to_error_code():
    assert http_status_to_error_code(400) == "BAD_REQUEST"
    assert http_status_to_error_code(404) == "NOT_FOUND"
    assert http_status_to_error_code(422) == "VALIDATION_ERROR"
    assert http_status_to_error_code(500) == "INTERNAL_SERVER_ERROR"
    assert http_status_to_error_code(502) == "MODEL_SERVICE_ERROR"
    assert http_status_to_error_code(504) == "MODEL_SERVICE_TIMEOUT"
    assert http_status_to_error_code(401) == "HTTP_ERROR"
