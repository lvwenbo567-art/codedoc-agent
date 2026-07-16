import json
import logging
from pathlib import Path
from typing import List


logger = logging.getLogger(__name__)


class JsonDataError(ValueError):
    """
    Raised when a JSON file is syntactically invalid or violates the expected
    business schema.
    """


class MissingFieldError(JsonDataError):
    """
    Raised when a JSON object misses required business fields.
    """


def read_json_list(
    file_path: str,
    required_fields: List[str] | None = None,
) -> List[dict]:
    """
    Read a UTF-8 JSON file whose root node must be a list of dictionaries.
    """
    path = Path(file_path)
    fields = required_fields or []

    if not path.exists():
        logger.error("JSON file does not exist: %s", file_path)
        raise FileNotFoundError(f"JSON file does not exist: {file_path}")

    if not path.is_file():
        logger.error("JSON path is not a file: %s", file_path)
        raise ValueError(f"JSON path is not a file: {file_path}")

    try:
        content = path.read_text(encoding="utf-8")

    except UnicodeDecodeError as exc:
        logger.error("JSON file encoding error: %s", file_path)
        raise UnicodeError(f"cannot read JSON file as UTF-8: {file_path}") from exc

    try:
        data = json.loads(content)

    except json.JSONDecodeError as exc:
        logger.error(
            "JSON syntax error: %s, line=%s, column=%s",
            file_path,
            exc.lineno,
            exc.colno,
        )
        raise JsonDataError(f"JSON syntax error at line {exc.lineno}") from exc

    if not isinstance(data, list):
        logger.warning("JSON root node is not a list: %s", file_path)
        raise JsonDataError("JSON root node must be a list")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning("JSON item is not an object: index=%s", index)
            raise JsonDataError(f"JSON item at index {index} must be an object")

        missing_fields = [
            field
            for field in fields
            if field not in item
        ]

        if missing_fields:
            logger.warning(
                "JSON item missing fields: index=%s, fields=%s",
                index,
                missing_fields,
            )
            raise MissingFieldError(
                f"JSON item at index {index} missing fields: {missing_fields}"
            )

    logger.info("JSON file loaded successfully: %s, count=%s", file_path, len(data))

    return data
