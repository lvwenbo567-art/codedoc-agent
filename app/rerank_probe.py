import argparse

from rerank_client import RerankClient, RerankConfig


def main() -> None:
    """
    命令行探针：加载真实 CrossEncoder Rerank 模型并输出简单打分结果。
    """
    parser = argparse.ArgumentParser(
        description="测试真实 Rerank 模型"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="模型名称或本地模型目录",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="推理设备，例如 cpu 或 cuda",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="只从本地目录加载模型，不联网下载",
    )
    args = parser.parse_args()

    config = RerankConfig(
        provider="sentence_transformers",
        model_name_or_path=args.model,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    client = RerankClient(config=config)
    query = "这个项目如何构建向量索引？"
    documents = [
        "项目通过 /index 接口读取 chunks 并生成向量。",
        "数据库中包含 projects、files 和 chunks 表。",
        "算法题使用二叉树进行练习。",
    ]
    scores = client.score(
        query=query,
        documents=documents,
    )

    for index, score in enumerate(scores, start=1):
        print(f"Document {index}: {score:.6f}")


if __name__ == "__main__":
    main()
