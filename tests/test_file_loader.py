from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from file_loader import scan_project_files, load_project_files

def test_scan_project_files_supported_suffixes(tmp_path):
    readme = tmp_path / "README.md"
    note = tmp_path / "note.txt"
    code = tmp_path / "main.py"
    image = tmp_path / "image.png"

    readme.write_text("# hello", encoding="utf-8")
    note.write_text("note", encoding="utf-8")
    code.write_text("def main():\n    pass", encoding="utf-8")
    image.write_text("fake image", encoding="utf-8")

    files = scan_project_files(str(tmp_path))
    suffixes = {path.suffix for path in files}

    assert ".md" in suffixes
    assert ".txt" in suffixes
    assert ".py" in suffixes
    assert ".png" not in suffixes

def test_load_project_files_schema(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# hello", encoding="utf-8")

    files = load_project_files(str(tmp_path))

    assert len(files) == 1

    file = files[0]

    assert "path" in file
    assert "name" in file
    assert "suffix" in file
    assert "content" in file
    assert "length" in file

    assert file["name"] == "README.md"
    assert file["suffix"] == ".md"
    assert file["content"] == "# hello"
    assert file["length"] == len("# hello")

def test_scan_project_files_not_exists():
    with pytest.raises(FileNotFoundError):
        scan_project_files("not_exists_path")

def test_scan_project_files_not_directory(tmp_path):
    file_path = tmp_path / "hello.txt"
    file_path.write_text("hello", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        scan_project_files(str(file_path))