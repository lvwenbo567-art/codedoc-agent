from pathlib import Path
from typing import Dict, List

SUPPORTED_SUFFIXES={".md",".txt",".py"}

def scan_project_files(project_path: str) -> List[Path]:
     """
    扫描项目目录，返回支持的文件路径列表。
    """
     root=Path(project_path)

     if not root.exists():
          raise FileNotFoundError(f"项目路径不存在：{project_path}")
     
     if not root.is_dir():
          raise NotADirectoryError(f"不是一个目录:{project_path}")
     
     files=[]

     for path in root.rglob("*"):
          if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
               files.append(path)


     return files

def read_text_file(file_path:Path) ->str:
      """
    读取文本文件内容，优先 utf-8，失败后尝试 gbk。
    """
      try:
           return file_path.read_text(encoding="utf-8")
      except UnicodeDecodeError:
           return file_path.read_text(encoding="gbk",errors="ignore")

def load_project_files(project_path: str) -> List[Dict]:
    """
    加载项目中的 .md / .txt / .py 文件。
    """
    file_paths = scan_project_files(project_path)

    results = []

    for file_path in file_paths:
        content = read_text_file(file_path)

        results.append(
            {
                "path": str(file_path),
                "name": file_path.name,
                "suffix": file_path.suffix.lower(),
                "content": content,
                "length": len(content),
            }
        )

    return results