from __future__ import annotations#延迟处理类型注解

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


CodeSymbolType = Literal[
    "module",
    "function",
    "async_function",
    "class",
    "method",
    "async_method",
]


class PythonCodeParseError(ValueError):
    """
    Python 源码无法通过 AST 解析时抛出的异常。
    """


@dataclass(frozen=True)
class CodeSymbol:
    """
    一个可检索的 Python 代码结构单元。
    """

    symbol_name: str
    qualified_name: str#完整名字
    symbol_type: CodeSymbolType
    parent_class: str | None
    signature: str#保存函数、类或者方法的签名
    start_line: int
    end_line: int
    docstring: str
    content: str

    def to_dict(self) -> dict:
        """
        转成普通字典，方便后续构造 chunk 和 JSON 序列化。
        """
        return asdict(self)


def _node_start_line(node: ast.AST) -> int:
    """
    如果函数或类有装饰器，从第一个装饰器所在行开始计算。
    """
    decorator_list = getattr(node, "decorator_list", [])#getattr(对象, "属性名", 默认值)
    '''
    如果 node 有 decorator_list 属性：
    decorator_list = node.decorator_list
否则：
    decorator_list = []
    '''
    if decorator_list:
        return min(decorator.lineno for decorator in decorator_list)

    return getattr(node, "lineno", 1)


def _node_end_line(node: ast.AST) -> int:
    """
    获取 AST 节点结束行，兼容少数没有 end_lineno 的情况。
    """
    return getattr(node, "end_lineno", None) or getattr(node, "lineno", 1)


def _extract_source_lines(
    source_lines: list[str],#source_lines = source.splitlines()
    start_line: int,
    end_line: int,
) -> str:
    """
    AST 行号从 1 开始，列表下标从 0 开始。
    """
    return "\n".join(source_lines[start_line - 1:end_line]).rstrip()#删除末尾空白


def _safe_unparse(node: ast.AST | None) -> str:
    """
    尽量把 AST 节点还原成源码文本，失败时返回空字符串。
    """
    if node is None:
        return ""

    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _build_function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    """
    构造函数或方法签名。
    """
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    arguments = _safe_unparse(node.args)#"a: int, b: int=0, *args, **kwargs"
    return_annotation = ""

    if node.returns is not None:
        return_annotation = f" -> {_safe_unparse(node.returns)}"#def add(a: int, b: int) -> int: int

    return f"{prefix} {node.name}({arguments}){return_annotation}"
    #"def add(a: int, b: int=0) -> int"

def _build_class_signature(node: ast.ClassDef) -> str:
    """
    构造类签名，包括继承类和关键字参数。
    """
    arguments: list[str] = []
    #class User(BaseModel, Serializable,metaclass=ABCMeta):
    for base in node.bases:
        base_text = _safe_unparse(base)
        if base_text:
            arguments.append(base_text)
            """
            arguments = [
                "BaseModel",
                "Serializable",
            ]
            """
    for keyword in node.keywords:
        value_text = _safe_unparse(keyword.value)
        if keyword.arg is None:
            arguments.append(f"**{value_text}")
        else:
            arguments.append(f"{keyword.arg}={value_text}")
        """
        [
            "BaseModel",
            "Serializable",
            "metaclass=ABCMeta",
        ]
        """
    if arguments:
        return f"class {node.name}({', '.join(arguments)})"
        #"class User(BaseModel, Serializable, metaclass=ABCMeta)"
    return f"class {node.name}"
      #"class Calculator"

def _build_class_summary(node: ast.ClassDef) -> str:
    """
    类 chunk 只保存类摘要，避免和方法 chunk 大量重复。
    """
    lines = [_build_class_signature(node)]
    docstring = ast.get_docstring(node, clean=False) or ""

    if docstring:
        lines.extend(["", "Docstring:", docstring])

    class_attributes: list[str] = []
    method_signatures: list[str] = []

    for child in node.body:
        if isinstance(child, (ast.Assign, ast.AnnAssign)):
            attribute = _safe_unparse(child)
            if attribute:
                class_attributes.append(attribute)

        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_signatures.append(_build_function_signature(child))

    if class_attributes:
        lines.extend(["", "Class attributes:"])
        lines.extend(f"- {attribute}" for attribute in class_attributes)

    if method_signatures:
        lines.extend(["", "Methods:"])
        lines.extend(f"- {signature}" for signature in method_signatures)

    return "\n".join(lines).strip()


def _build_module_symbol(
    tree: ast.Module,
    source_lines: list[str],
) -> CodeSymbol | None:
    """
    提取模块级导入、常量和普通语句；函数和类会单独形成 chunk。
    """
    module_nodes: list[ast.AST] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        module_nodes.append(node)

    if not module_nodes:
        return None

    source_parts: list[str] = []

    for node in module_nodes:
        part = _extract_source_lines(
            source_lines=source_lines,
            start_line=_node_start_line(node),
            end_line=_node_end_line(node),
        )
        if part.strip():
            source_parts.append(part)

    if not source_parts:
        return None

    return CodeSymbol(
        symbol_name="__module__",
        qualified_name="__module__",
        symbol_type="module",
        parent_class=None,
        signature="module",
        start_line=min(_node_start_line(node) for node in module_nodes),
        end_line=max(_node_end_line(node) for node in module_nodes),
        docstring=ast.get_docstring(tree, clean=False) or "",
        content="\n\n".join(source_parts),
    )


def parse_python_symbols(
    source: str,
    source_path: str = "<memory>",
) -> list[dict]:
    """
    将 Python 源码解析为模块、函数、类和方法级结构。
    """
    if not isinstance(source, str):
        raise TypeError("source 必须是字符串")

    try:
        tree = ast.parse(source, filename=source_path)
    except SyntaxError as exc:
        raise PythonCodeParseError(
            f"Python AST 解析失败：{source_path}：第 {exc.lineno} 行，{exc.msg}"
        ) from exc

    source_lines = source.splitlines()
    symbols: list[CodeSymbol] = []

    module_symbol = _build_module_symbol(tree=tree, source_lines=source_lines)
    if module_symbol is not None:
        symbols.append(module_symbol)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start_line = _node_start_line(node)
            end_line = _node_end_line(node)
            symbol_type: CodeSymbolType = (
                "async_function"
                if isinstance(node, ast.AsyncFunctionDef)
                else "function"
            )
            symbols.append(
                CodeSymbol(
                    symbol_name=node.name,
                    qualified_name=node.name,
                    symbol_type=symbol_type,
                    parent_class=None,
                    signature=_build_function_signature(node),
                    start_line=start_line,
                    end_line=end_line,
                    docstring=ast.get_docstring(node, clean=False) or "",
                    content=_extract_source_lines(source_lines, start_line, end_line),
                )
            )

        elif isinstance(node, ast.ClassDef):
            class_start = _node_start_line(node)
            class_end = _node_end_line(node)
            symbols.append(
                CodeSymbol(
                    symbol_name=node.name,
                    qualified_name=node.name,
                    symbol_type="class",
                    parent_class=None,
                    signature=_build_class_signature(node),
                    start_line=class_start,
                    end_line=class_end,
                    docstring=ast.get_docstring(node, clean=False) or "",
                    content=_build_class_summary(node),
                )
            )

            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                start_line = _node_start_line(child)
                end_line = _node_end_line(child)
                method_type: CodeSymbolType = (
                    "async_method"
                    if isinstance(child, ast.AsyncFunctionDef)
                    else "method"
                )
                symbols.append(
                    CodeSymbol(
                        symbol_name=child.name,
                        qualified_name=f"{node.name}.{child.name}",
                        symbol_type=method_type,
                        parent_class=node.name,
                        signature=_build_function_signature(child),
                        start_line=start_line,
                        end_line=end_line,
                        docstring=ast.get_docstring(child, clean=False) or "",
                        content=_extract_source_lines(source_lines, start_line, end_line),
                    )
                )

    return [symbol.to_dict() for symbol in symbols]


def parse_python_file(file_path: str) -> list[dict]:
    """
    从真实 Python 文件读取源码并解析结构。
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Python 文件不存在：{file_path}")

    if not path.is_file():
        raise ValueError(f"路径不是文件：{file_path}")

    return parse_python_symbols(
        source=path.read_text(encoding="utf-8"),
        source_path=str(path),
    )


def parse_python_code(content: str, file_path: str) -> dict:
    """
    兼容早期 Day2 测试的解析入口。
    """
    try:
        symbols = parse_python_symbols(source=content, source_path=file_path)
    except PythonCodeParseError as exc:
        return {
            "file_path": file_path,
            "functions": [],
            "classes": [],
            "error": f"SyntaxError: {exc}",
        }

    functions = []
    classes = []

    for symbol in symbols:
        symbol_type = symbol["symbol_type"]

        if symbol_type in {"function", "async_function", "method", "async_method"}:
            item = {
                "name": symbol["symbol_name"],
                "lineno": symbol["start_line"],
                "docstring": symbol["docstring"] or None,
            }
            if symbol_type in {"async_function", "async_method"}:
                item["is_async"] = True
            functions.append(item)

        elif symbol_type == "class":
            classes.append(
                {
                    "name": symbol["symbol_name"],
                    "lineno": symbol["start_line"],
                    "docstring": symbol["docstring"] or None,
                }
            )

    return {
        "file_path": file_path,
        "functions": functions,
        "classes": classes,
        "error": None,
    }


def parse_python_files(files: list[dict]) -> list[dict]:
    """
    兼容早期入口：从项目文件列表中筛选 Python 文件并解析。
    """
    results = []

    for file in files:
        if file.get("suffix") != ".py":
            continue

        results.append(
            parse_python_code(
                content=file["content"],
                file_path=file["path"],
            )
        )

    return results
