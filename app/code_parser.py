import ast
from typing import Dict, List


def parse_python_code(content: str, file_path: str) -> Dict:
    """
    解析 Python 源代码，提取函数、类和 docstring 信息。
    """
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {
            "file_path": file_path,
            "functions": [],
            "classes": [],
            "error": f"SyntaxError: {e}",
        }

    functions = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(
                {
                    "name": node.name,
                    "lineno": node.lineno,
                    "docstring": ast.get_docstring(node),
                }
            )

        elif isinstance(node, ast.AsyncFunctionDef):
            functions.append(
                {
                    "name": node.name,
                    "lineno": node.lineno,
                    "docstring": ast.get_docstring(node),
                    "is_async": True,
                }
            )

        elif isinstance(node, ast.ClassDef):
            classes.append(
                {
                    "name": node.name,
                    "lineno": node.lineno,
                    "docstring": ast.get_docstring(node),
                }
            )

    return {
        "file_path": file_path,
        "functions": functions,
        "classes": classes,
        "error": None,
    }


def parse_python_files(files: List[Dict]) -> List[Dict]:
    """
    从项目文件列表中筛选 Python 文件，并解析其中的函数和类。
    """
    python_files = [f for f in files if f["suffix"] == ".py"]

    results = []

    for file in python_files:
        result = parse_python_code(
            content=file["content"],
            file_path=file["path"],
        )
        results.append(result)

    return results