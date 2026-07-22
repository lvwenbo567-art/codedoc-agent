from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from ingestion.code_parser import parse_python_code


def test_parse_python_code_function_and_class():
    code = '''
class Calculator:
    """简单计算器类。"""

    def add(self, a, b):
        """返回两个数的和。"""
        return a + b


def main():
    pass
'''

    result = parse_python_code(
        content=code,
        file_path="test_project/utils.py",
    )

    function_names = [func["name"] for func in result["functions"]]
    class_names = [cls["name"] for cls in result["classes"]]

    assert result["error"] is None
    assert "Calculator" in class_names
    assert "add" in function_names
    assert "main" in function_names