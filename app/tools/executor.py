from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic import ValidationError

from tools.errors import ToolBusinessError
from tools.models import ToolResult
from tools.registry import ToolRegistry


logger = logging.getLogger(__name__)
'''
1. registry.get()
   查找工具并检查白名单

2. _parse_arguments()
   把JSON字符串转换成字典

3. model_validate()
   用Pydantic校验参数

4. model_dump() + **
   把校验后的参数传给handler

5. try...except
   区分业务错误和未知错误

6. ToolResult
   统一成功和失败返回格式
'''

class ToolExecutor:
    """
    统一工具执行器。

    职责：
    1. 检查工具是否在白名单
    2. 解析 JSON 参数
    3. 使用 Pydantic 校验参数
    4. 执行 handler
    5. 将成功和失败统一成 ToolResult
    """

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:
        self.registry = registry

    def execute(
        self,
        tool_name: str,
        arguments: str | dict[str, Any],
    ) -> ToolResult:
        start_time = time.perf_counter()

        tool = self.registry.get(tool_name)

        if tool is None:
            return self._error_result(
                tool_name=tool_name,
                error_code="TOOL_NOT_FOUND",
                error_message=(
                    f"工具未注册或不在白名单中："
                    f"{tool_name}"
                ),
                start_time=start_time,
            )

        try:
            parsed_arguments = (
                self._parse_arguments(arguments)
            )

        except ValueError as exc:
            return self._error_result(
                tool_name=tool_name,
                error_code="INVALID_JSON",
                error_message=str(exc),
                start_time=start_time,
            )

        try:
            validated_arguments = (
                tool.args_model.model_validate(
                    parsed_arguments
                )
            )#Pydantic校验并补充默认值

        except ValidationError as exc:
            errors = exc.errors(
                include_url=False
            )

            return self._error_result(
                tool_name=tool_name,
                error_code="INVALID_ARGUMENTS",
                error_message=json.dumps(
                    errors,
                    ensure_ascii=False,
                ),
                start_time=start_time,
            )

        try:
            result_data = tool.handler(
                **validated_arguments.model_dump()
            )#转换成字典并展开

            if isinstance(
                result_data,
                ToolResult,
            ):
                return result_data

            duration_ms = self._duration_ms(
                start_time
            )

            return ToolResult(
                success=True,
                tool_name=tool_name,
                data=result_data,
                duration_ms=duration_ms,
            )

        except ToolBusinessError as exc:
            return self._error_result(
                tool_name=tool_name,
                error_code=exc.error_code,
                error_message=exc.message,
                start_time=start_time,
            )

        except Exception as exc:
            logger.exception(
                "工具执行失败：tool=%s",
                tool_name,
            )

            return self._error_result(
                tool_name=tool_name,
                error_code=(
                    "TOOL_EXECUTION_ERROR"
                ),
                error_message=str(exc),
                start_time=start_time,
            )

    @staticmethod
    def _parse_arguments(
        arguments: str | dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments

        if not isinstance(arguments, str):
            raise ValueError(
                "工具参数必须是 JSON 字符串"
                "或字典"
            )

        value = arguments.strip()

        if not value:
            return {}

        try:
            parsed = json.loads(value)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "工具参数不是合法 JSON："
                f"{exc.msg}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "工具参数 JSON 顶层必须是对象"
            )

        return parsed

    @staticmethod
    def _duration_ms(
        start_time: float,
    ) -> float:
        return round(
            (
                time.perf_counter()
                - start_time
            )
            * 1000,
            2,
        )

    def _error_result(
        self,
        tool_name: str,
        error_code: str,
        error_message: str,
        start_time: float,
    ) -> ToolResult:
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error_code=error_code,
            error_message=error_message,
            duration_ms=self._duration_ms(
                start_time
            ),
        )
