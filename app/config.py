import os

SUPPORTED_SUFFIXES = {".md", ".txt", ".py"}

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

DEFAULT_MODEL_NAME = "mock-model"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"

DEFAULT_DB_PATH = "data/codedoc.db"

DEFAULT_EMBEDDING_MODEL = "mock-hash-embedding"
DEFAULT_EMBEDDING_DIMENSION = 64
DEFAULT_VECTOR_INDEX_PATH = "outputs/vector_index.json"

DEFAULT_CHAT_MODEL = "mock-chat-model"
DEFAULT_MAX_CONTEXT_CHARS = 6000
DEFAULT_RAG_TOP_K = 5

DEFAULT_CHAT_PROVIDER = os.getenv(
    "CODEDOC_CHAT_PROVIDER",
    "mock",
)

DEFAULT_CHAT_MODEL = os.getenv(
    "CODEDOC_CHAT_MODEL",
    "mock-chat-model",
)

DEFAULT_CHAT_BASE_URL = os.getenv(
    "CODEDOC_CHAT_BASE_URL",
    "http://localhost:11434/v1",
)

DEFAULT_CHAT_API_KEY = os.getenv(
    "CODEDOC_CHAT_API_KEY",
    "EMPTY",
)

DEFAULT_CHAT_TIMEOUT_SECONDS = float(
    os.getenv(
        "CODEDOC_CHAT_TIMEOUT_SECONDS",
        "30",
    )
)

DEFAULT_CHAT_TEMPERATURE = float(
    os.getenv(
        "CODEDOC_CHAT_TEMPERATURE",
        "0.2",
    )
)

DEFAULT_CHAT_MAX_TOKENS = int(
    os.getenv(
        "CODEDOC_CHAT_MAX_TOKENS",
        "800",
    )
)
DEFAULT_EMBEDDING_PROVIDER = os.getenv(
    "CODEDOC_EMBEDDING_PROVIDER",
    "mock",
)

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "CODEDOC_EMBEDDING_MODEL",
    "mock-hash-embedding",
)

DEFAULT_EMBEDDING_BASE_URL = os.getenv(
    "CODEDOC_EMBEDDING_BASE_URL",
    "http://localhost:11434",
)

DEFAULT_EMBEDDING_API_KEY = os.getenv(
    "CODEDOC_EMBEDDING_API_KEY",
    "",
)

DEFAULT_EMBEDDING_TIMEOUT_SECONDS = float(
    os.getenv(
        "CODEDOC_EMBEDDING_TIMEOUT_SECONDS",
        "30",
    )
)

DEFAULT_EMBEDDING_DIMENSION = int(
    os.getenv(
        "CODEDOC_EMBEDDING_DIMENSION",
        "64",
    )
)

DEFAULT_NORMALIZE_EMBEDDING = (
    os.getenv(
        "CODEDOC_NORMALIZE_EMBEDDING",
        "true",
    ).lower()
    == "true"
)

DEFAULT_VECTOR_INDEX_PATH = os.getenv(
    "CODEDOC_VECTOR_INDEX_PATH",
    "outputs/vector_index.json",
)

VECTOR_INDEX_FORMAT_VERSION = "1.0"

DEFAULT_EMBEDDING_BATCH_SIZE = int(
    os.getenv(
        "CODEDOC_EMBEDDING_BATCH_SIZE",
        "16",
    )
)

DEFAULT_EMBEDDING_MAX_RETRIES = int(
    os.getenv(
        "CODEDOC_EMBEDDING_MAX_RETRIES",
        "2",
    )
)

DEFAULT_EMBEDDING_RETRY_BACKOFF_SECONDS = float(
    os.getenv(
        "CODEDOC_EMBEDDING_RETRY_BACKOFF_SECONDS",
        "1",
    )
)