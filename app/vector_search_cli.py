import argparse

from logger import setup_logger
from vector_search_service import search_vector_index_from_file


def main() -> None:
    logger = setup_logger()
    logger.info("开始运行向量检索 CLI")

    parser = argparse.ArgumentParser(
        description="从向量索引文件中检索相关 chunks"
    )

    parser.add_argument(
        "--index_path",
        default="outputs/vector_index.json",
        help="向量索引文件路径",
    )

    parser.add_argument(
        "--query",
        required=True,
        help="用户检索问题",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="返回的检索结果数量",
    )

    parser.add_argument(
        "--model_name",
        default="mock-hash-embedding",
        help="Embedding 模型名称",
    )

    parser.add_argument(
        "--dimension",
        type=int,
        default=64,
        help="Embedding 向量维度",
    )

    parser.add_argument(
        "--chunk_type",
        default=None,
        help="可选的 chunk 类型过滤，例如 code 或 document",
    )

    args = parser.parse_args()

    result = search_vector_index_from_file(
        query=args.query,
        index_path=args.index_path,
        top_k=args.top_k,
        model_name=args.model_name,
        dimension=args.dimension,
        chunk_type=args.chunk_type,
    )

    print("向量检索结果:")
    print(f"- query: {result['query']}")
    print(f"- index_path: {result['index_path']}")
    print(f"- top_k: {result['top_k']}")
    print(f"- result_count: {result['result_count']}")
    print()

    for item in result["results"]:
        print(f"Rank {item['rank']}")
        print(f"- score: {item['score']:.4f}")
        print(f"- chunk_id: {item['chunk_id']}")
        print(f"- source_name: {item['source_name']}")
        print(f"- chunk_type: {item['chunk_type']}")
        print(f"- content_preview: {item['content_preview']}")
        print()

    logger.info(
        "向量检索结束，命中数量=%s",
        result["result_count"],
    )


if __name__ == "__main__":
    main()