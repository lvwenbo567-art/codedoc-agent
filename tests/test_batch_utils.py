from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from batch_utils import split_batches


def test_split_batches():
    result = split_batches(
        [1, 2, 3, 4, 5],
        batch_size=2,
    )

    assert result == [
        [1, 2],
        [3, 4],
        [5],
    ]


def test_split_empty_list():
    assert split_batches(
        [],
        batch_size=2,
    ) == []


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        split_batches(
            [1, 2],
            batch_size=0,
        )
