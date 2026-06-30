import argparse

from file_loader import load_project_files
from code_parser import parse_python_files
from config import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL_NAME
from llm_client import LLMClient
from chunker import build_chunks
from logger import setup_logger

from config import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
)

def print_project_summary(
    project_path: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> None:
    logger = setup_logger()

    logger.info("开始读取项目文件: %s", project_path)
    files = load_project_files(project_path)
    logger.info("项目文件读取完成，文件数量: %s", len(files))

    logger.info("开始构建 chunks")
    chunks = build_chunks(
    files=files,
    chunk_size=chunk_size,
    overlap=overlap,
)
    logger.info("Chunk 构建完成，chunk 数量: %s", len(chunks))

    md_files = [f for f in files if f["suffix"] == ".md"]
    txt_files = [f for f in files if f["suffix"] == ".txt"]
    py_files = [f for f in files if f["suffix"] == ".py"]

    print(f"项目路径: {project_path}")
    print(f"共读取文件: {len(files)} 个")
    print()

    doc_chunks = [c for c in chunks if c["chunk_type"] == "document"]
    code_chunks = [c for c in chunks if c["chunk_type"] == "code"]

    print("Chunk 统计:")
    print(f"- 文档 chunk 数量: {len(doc_chunks)}")
    print(f"- 代码 chunk 数量: {len(code_chunks)}")
    print(f"- 总 chunk 数量: {len(chunks)}")
    print()
    

    print("Chunk 示例:")

    if chunks:
        for chunk in chunks[:3]:
            print(f"- chunk_id: {chunk['chunk_id']}")
            print(f"  source_name: {chunk['source_name']}")
            print(f"  chunk_type: {chunk['chunk_type']}")
            print(f"  content_preview: {chunk['content'][:100]}")
            print()
    else:
        print("- 暂无 chunk")
        print()
          
    logger.info(
        "Chunk 统计完成，document=%s, code=%s, total=%s",
        len(doc_chunks),
        len(code_chunks),
        len(chunks),
    )

    print("Markdown 文件:")
    for f in md_files:
        print(f"- {f['name']} ({f['length']} 字符)")
    print()

    print("TXT 文件:")
    for f in txt_files:
        print(f"- {f['name']} ({f['length']} 字符)")
    print()

    print("Python 文件:")
    for f in py_files:
        print(f"- {f['name']} ({f['length']} 字符)")
    print()

    readme = next((f for f in md_files if f["name"].lower() == "readme.md"), None)

    if readme:
        logger.info("找到 README.md，开始生成模拟摘要")

        print("README 前 500 字:")
        print(readme["content"][:500])

        llm_client = LLMClient(
            model_name=DEFAULT_MODEL_NAME,
            base_url=DEFAULT_BASE_URL,
            api_key=DEFAULT_API_KEY,
        )

        summary = llm_client.summarize_text(readme["content"])

        print()
        print("README 模拟摘要:")
        print(summary)

        logger.info("README 模拟摘要生成完成")
    else:
        logger.warning("未找到 README.md")
        print("未找到 README.md")

    print()
    print("Python 代码结构:")

    logger.info("开始解析 Python 文件结构")
    python_structures = parse_python_files(files)
    logger.info("Python 文件结构解析完成，解析文件数量: %s", len(python_structures))

    for item in python_structures:
        print(f"\n文件: {item['file_path']}")

        if item["error"]:
            logger.error("Python 文件解析失败: %s, error=%s", item["file_path"], item["error"])
            print(f"  解析失败: {item['error']}")
            continue

        print("  类:")
        if item["classes"]:
            for cls in item["classes"]:
                print(f"  - {cls['name']}，第 {cls['lineno']} 行")
                if cls["docstring"]:
                    print(f"    注释: {cls['docstring']}")
        else:
            print("  - 无")

        print("  函数:")
        if item["functions"]:
            for func in item["functions"]:
                async_flag = "async " if func.get("is_async") else ""
                print(f"  - {async_flag}{func['name']}，第 {func['lineno']} 行")
                if func["docstring"]:
                    print(f"    注释: {func['docstring']}")
        else:
            print("  - 无")


def main() -> None:
    logger = setup_logger()
    logger.info("开始运行 CodeDoc Research Agent")

    parser = argparse.ArgumentParser(description="CodeDoc Research Agent 项目扫描器")
    parser.add_argument("--project_path", required=True, help="要分析的项目目录路径")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="每个 chunk 的最大字符数")
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="相邻 chunk 的重叠字符数")
    args = parser.parse_args()

    logger.info("命令行参数解析完成，project_path=%s", args.project_path)

    print_project_summary(
    project_path=args.project_path,
    chunk_size=args.chunk_size,
    overlap=args.overlap,
)

    logger.info("CodeDoc Research Agent 运行结束")


if __name__ == "__main__":
    main()