import argparse

from logger import setup_logger
from search_service import search_chunks_from_json

def main()->None:
    logger=setup_logger()
    logger.info("开始运行 chunks 检索 CLI")
    
    parser=argparse.ArgumentParser(description="从 chunks.json 中检索相关 chunks")

    parser.add_argument(
        "--chunks_path",
        default="outputs/chunks.json",
        help="chunks JSON 文件路径",
    )

    parser.add_argument(
        "--query",
        required=True,
        help="检索查询关键词",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="返回的检索结果数量",
    )
    args=parser.parse_args()

    logger.info(
        "检索参数解析完成，chunks_path=%s, query=%s, top_k=%s",
        args.chunks_path,
        args.query,
        args.top_k,
    )

    results = search_chunks_from_json(
        input_path=args.chunks_path,
        query=args.query,
        top_k=args.top_k,
    )

    print("检索结果:")
    print(f"- chunks_path: {args.chunks_path}")
    print(f"- query: {args.query}")
    print(f"- top_k: {args.top_k}")
    print(f"- 命中数量: {len(results)}")
    print()

    if results:
        for item in results:
            print(f"Rank {item['rank']}")
            print(f"- score: {item['score']}")
            print(f"- chunk_id: {item['chunk_id']}")
            print(f"- source_name: {item['source_name']}")
            print(f"- chunk_type: {item['chunk_type']}")
            print(f"- content_preview: {item['content_preview']}")
            print()
    else:
        print("- 未检索到相关 chunk")
        print()

    logger.info("chunks 检索 CLI 运行结束，命中数量: %s", len(results))


if __name__ == "__main__":
    main()