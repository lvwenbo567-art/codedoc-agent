import json
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from day23_json_reader import JsonDataError, MissingFieldError, read_json_list


def test_read_json_list_success(tmp_path):
    json_path = tmp_path / "data.json"
    json_path.write_text(
        json.dumps(
            [
                {"id": 1, "name": "CodeDoc"},
                {"id": 2, "name": "RAG"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    data = read_json_list(
        str(json_path),
        required_fields=["id", "name"],
    )

    assert len(data) == 2
    assert data[0]["name"] == "CodeDoc"


def test_read_json_list_file_not_exists(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_json_list(str(tmp_path / "missing.json"))


def test_read_json_list_invalid_json(tmp_path):
    json_path = tmp_path / "bad.json"
    json_path.write_text("[", encoding="utf-8")

    with pytest.raises(JsonDataError, match="JSON syntax error"):
        read_json_list(str(json_path))


def test_read_json_list_root_must_be_list(tmp_path):
    json_path = tmp_path / "object.json"
    json_path.write_text('{"id": 1}', encoding="utf-8")

    with pytest.raises(JsonDataError, match="root node must be a list"):
        read_json_list(str(json_path))


def test_read_json_list_items_must_be_dict(tmp_path):
    json_path = tmp_path / "items.json"
    json_path.write_text('[{"id": 1}, 2]', encoding="utf-8")

    with pytest.raises(JsonDataError, match="index 1"):
        read_json_list(str(json_path))


def test_read_json_list_required_fields(tmp_path):
    json_path = tmp_path / "missing_field.json"
    json_path.write_text('[{"id": 1}]', encoding="utf-8")

    with pytest.raises(MissingFieldError, match="missing fields"):
        read_json_list(
            str(json_path),
            required_fields=["id", "name"],
        )
