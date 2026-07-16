import json
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from day24_file_config import (
    PracticeConfig,
    export_config,
    load_practice_config,
    read_json_file,
    save_json_file,
)


def test_read_json_file_success(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"embedding_model": "bge-m3"}),
        encoding="utf-8",
    )

    data = read_json_file(str(config_path))

    assert data["embedding_model"] == "bge-m3"


def test_read_json_file_not_exists(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_json_file(str(tmp_path / "missing.json"))


def test_read_json_file_path_is_not_file(tmp_path):
    with pytest.raises(ValueError, match="不是文件"):
        read_json_file(str(tmp_path))


def test_read_json_file_root_must_be_dict(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="根节点必须是对象"):
        read_json_file(str(config_path))


def test_save_json_file_creates_parent_dir(tmp_path):
    output_path = tmp_path / "nested" / "config.json"

    saved_path = save_json_file(
        data={"embedding_model": "mock-hash-embedding"},
        output_path=str(output_path),
    )

    assert saved_path == output_path
    assert output_path.exists()
    assert read_json_file(str(output_path))["embedding_model"] == "mock-hash-embedding"


def test_load_practice_config_env_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "embedding_model": "mock-hash-embedding",
                "database_path": "data/codedoc.db",
                "output_dir": "outputs",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PRACTICE_EMBEDDING_MODEL", "bge-m3")

    config = load_practice_config(str(config_path))

    assert config.embedding_model == "bge-m3"
    assert config.database_path == "data/codedoc.db"
    assert config.output_dir == "outputs"


def test_export_config(tmp_path):
    output_path = tmp_path / "export" / "config.json"

    saved_path = export_config(
        PracticeConfig(
            embedding_model="bge-m3",
            database_path="data/codedoc.db",
            output_dir="outputs",
        ),
        str(output_path),
    )

    assert saved_path.exists()
    assert read_json_file(str(output_path))["embedding_model"] == "bge-m3"
