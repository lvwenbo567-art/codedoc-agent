import argparse
import json

from config import (
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_NORMALIZE_EMBEDDING,
)
from clients.embedding_client import EmbeddingClient, EmbeddingConfig


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器，用于单独测试 Embedding 服务。
    """
    parser = argparse.ArgumentParser(
        description="测试当前配置的 Embedding Provider 是否可用。",
    )

    parser.add_argument("--text", default="这个项目如何启动？")
    parser.add_argument("--provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_EMBEDDING_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_EMBEDDING_API_KEY)
    parser.add_argument("--timeout", type=float, default=DEFAULT_EMBEDDING_TIMEOUT_SECONDS)
    parser.add_argument("--mock-dimension", type=int, default=DEFAULT_EMBEDDING_DIMENSION)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="关闭 Embedding 归一化。",
    )

    return parser


def main() -> None:
    """
    读取命令行参数，调用 EmbeddingClient 并打印向量维度和前几个值。
    """
    args = build_parser().parse_args()

    config = EmbeddingConfig(
        provider=args.provider,
        model_name=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_seconds=args.timeout,
        mock_dimension=args.mock_dimension,
        normalize=DEFAULT_NORMALIZE_EMBEDDING and not args.no_normalize,
    )

    client = EmbeddingClient(config=config)
    embedding = client.embed_text(args.text)

    print(
        json.dumps(
            {
                "provider": config.provider,
                "model": config.model_name,
                "dimension": len(embedding),
                "preview": embedding[:5],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
