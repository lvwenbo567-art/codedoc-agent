from pathlib import Path
# =========================
# LLM 配置
# =========================
DEFAULT_MODEL_NAME = "mock-model"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"

# =========================
# 文件扫描配置
# =========================

SUPPORTED_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
}

MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024

# =========================
# Chunk 配置
# =========================

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

def normalize_project_path(project_path: str) -> Path:
    path = Path(project_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"项目路径不存在: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"项目路径不是目录: {path}")

    return path