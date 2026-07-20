import json
import os
from pathlib import Path
from typing import Any


def read_json_file(path: str) -> dict[str, Any]:
    """
    读取 JSON 文件，并要求文件根对象必须是 dict。
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON 文件不存在：{path}")

    data = json.loads(file_path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("JSON 根对象必须是字典")

    return data


def save_json_file(data: dict[str, Any], path: str) -> Path:
    """
    将字典保存为 JSON 文件，并自动创建父目录。
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return file_path


def load_config_with_env(
    config_path: str,
    env_prefix: str = "PRACTICE",
) -> dict[str, Any]:
    """
    读取配置文件，并用指定前缀的环境变量覆盖同名配置项。
    """
    config = read_json_file(config_path)

    for key in list(config.keys()):
        env_key = f"{env_prefix}_{key}".upper()

        if env_key in os.environ:
            config[key] = os.environ[env_key]

    return config
