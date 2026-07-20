from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from file_utils import load_config_with_env, read_json_file, save_json_file


def test_save_and_read_json_file(tmp_path):
    path = tmp_path / "nested" / "config.json"

    saved_path = save_json_file({"model": "mock"}, str(path))

    assert saved_path == path
    assert read_json_file(str(path)) == {"model": "mock"}


def test_read_json_file_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_json_file(str(tmp_path / "missing.json"))


def test_read_json_file_rejects_non_dict_root(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError):
        read_json_file(str(path))


def test_read_json_file_rejects_invalid_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError):
        read_json_file(str(path))


def test_load_config_with_env_overrides_existing_key(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    save_json_file({"model": "mock", "top_k": 3}, str(path))
    monkeypatch.setenv("PRACTICE_MODEL", "bge-m3")

    config = load_config_with_env(str(path))

    assert config["model"] == "bge-m3"
    assert config["top_k"] == 3
