import argparse

from file_loader import load_project_files
from code_parser import parse_python_files


def print_project_summary(project_path: str) -> None:
    files = load_project_files(project_path)

    md_files = [f for f in files if f["suffix"] == ".md"]
    txt_files = [f for f in files if f["suffix"] == ".txt"]
    py_files = [f for f in files if f["suffix"] == ".py"]

    print(f"项目路径: {project_path}")
    print(f"共读取文件: {len(files)} 个")
    print()

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
        print("README 前 500 字:")
        print(readme["content"][:500])
    else:
        print("未找到 README.md")

    print()
    print("Python 代码结构:")
    python_structures = parse_python_files(files)

    for item in python_structures:
        print(f"\n文件: {item['file_path']}")

        if item["error"]:
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
    parser = argparse.ArgumentParser(description="CodeDoc Research Agent 项目扫描器")
    parser.add_argument("--project_path", required=True, help="要分析的项目目录路径")

    args = parser.parse_args()
    print_project_summary(args.project_path)


if __name__ == "__main__":
    main()