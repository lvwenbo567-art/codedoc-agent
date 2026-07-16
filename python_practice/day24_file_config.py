import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PracticeConfig:
    """
    练习用配置对象，集中保存模型名、数据库路径和输出目录。
    """

    embedding_model: str
    database_path: str
    output_dir: str


def read_json_file(
    file_path: str,
) -> dict:
    """
    读取 JSON 配置文件，并要求根节点必须是对象。
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{file_path}")

    if not path.is_file():
        raise ValueError(f"输入路径不是文件：{file_path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("配置文件根节点必须是对象")

    return data


def save_json_file(
    data: dict,
    output_path: str,
) -> Path:
    """
    保存 JSON 文件，并自动创建不存在的父目录。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return path


def load_practice_config(
    config_path: str,
) -> PracticeConfig:
    """
    读取配置文件，并允许环境变量覆盖 JSON 中的默认值。
    """
    data = read_json_file(config_path)

    embedding_model = os.getenv(
        "PRACTICE_EMBEDDING_MODEL",
        data.get("embedding_model", "mock-hash-embedding"),
    )
    database_path = os.getenv(
        "PRACTICE_DATABASE_PATH",
        data.get("database_path", "data/codedoc.db"),
    )
    output_dir = os.getenv(
        "PRACTICE_OUTPUT_DIR",
        data.get("output_dir", "outputs"),
    )

    return PracticeConfig(
        embedding_model=embedding_model,
        database_path=database_path,
        output_dir=output_dir,
    )


def export_config(
    config: PracticeConfig,
    output_path: str,
) -> Path:
    """
    将 PracticeConfig 导出为 JSON 文件。
    """
    return save_json_file(
        data=asdict(config),
        output_path=output_path,
    )
