from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from processor import TextProcessor


def test_clean_text_collapses_whitespace():
    processor = TextProcessor()

    assert processor.clean_text("  hello\n\nworld\t!  ") == "hello world !"


def test_process_returns_result_without_truncation():
    processor = TextProcessor(max_length=20)
    result = processor.process("  hello   world  ")

    assert result.original_text == "  hello   world  "
    assert result.cleaned_text == "hello world"
    assert result.length == 11
    assert result.truncated is False


def test_process_truncates_long_text():
    processor = TextProcessor(max_length=5)
    result = processor.process("hello world")

    assert result.cleaned_text == "hello"
    assert result.length == 5
    assert result.truncated is True


def test_text_processor_rejects_invalid_max_length():
    with pytest.raises(ValueError):
        TextProcessor(max_length=0)
