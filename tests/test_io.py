"""Tests for the long-format CSV reader.

Every case here is a way a real sales export breaks: a bad number, an empty cell,
a wrong column name, several rows for one (store, week), a missing week.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecast.io import DataError, read_csv  # noqa: E402


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sales.csv"
    path.write_text(text, encoding="utf-8")
    return path


class TestReadCsv:
    def test_builds_a_hierarchy_and_sums_leaves_to_the_root(self, tmp_path):
        path = _write(
            tmp_path,
            "region,store,week,sales\n"
            "north,lahore,2024-01-01,120\n"
            "south,karachi,2024-01-01,210\n"
            "north,lahore,2024-01-08,100\n"
            "south,karachi,2024-01-08,200\n",
        )
        ds = read_csv(path, time="week", value="sales", levels=["region", "store"])
        assert ds.rows == 4
        assert ds.periods == ["2024-01-01", "2024-01-08"]
        assert ds.history["total"] == [330.0, 300.0]
        assert ds.history["north/lahore"] == [120.0, 100.0]
        assert set(ds.hierarchy.nodes) == {
            "total",
            "north",
            "south",
            "north/lahore",
            "south/karachi",
        }

    def test_missing_leaf_period_is_filled_with_zero_by_default(self, tmp_path):
        path = _write(
            tmp_path,
            "store,week,sales\nlahore,2024-01-01,120\nkarachi,2024-01-01,200\n"
            "lahore,2024-01-08,100\n",  # karachi has no week-2 row
        )
        ds = read_csv(path, time="week", value="sales", levels=["store"])
        assert ds.filled == 1
        assert ds.history["karachi"] == [200.0, 0.0]

    def test_missing_error_mode_refuses_instead_of_filling(self, tmp_path):
        path = _write(
            tmp_path,
            "store,week,sales\nlahore,2024-01-01,120\nkarachi,2024-01-01,200\n"
            "lahore,2024-01-08,100\n",
        )
        with pytest.raises(DataError, match="karachi"):
            read_csv(path, time="week", value="sales", levels=["store"], missing="error")

    def test_several_rows_for_the_same_cell_are_summed(self, tmp_path):
        path = _write(
            tmp_path,
            "store,week,sales\nlahore,2024-01-01,10\nlahore,2024-01-01,5\n",
        )
        ds = read_csv(path, time="week", value="sales", levels=["store"])
        assert ds.duplicates == 1
        assert ds.history["lahore"] == [15.0]

    def test_unknown_column_names_the_column_and_lists_real_ones(self, tmp_path):
        path = _write(tmp_path, "store,week,sales\nlahore,2024-01-01,10\n")
        with pytest.raises(DataError, match="'units' not found; columns are store, week, sales"):
            read_csv(path, time="week", value="units", levels=["store"])

    def test_non_numeric_value_names_the_line(self, tmp_path):
        path = _write(tmp_path, "store,week,sales\nlahore,2024-01-01,10\nlahore,2024-01-08,N/A\n")
        with pytest.raises(DataError, match=r":3: sales='N/A'"):
            read_csv(path, time="week", value="sales", levels=["store"])

    def test_nan_or_infinite_value_is_refused(self, tmp_path):
        path = _write(tmp_path, "store,week,sales\nlahore,2024-01-01,nan\n")
        with pytest.raises(DataError, match="fill missing values first"):
            read_csv(path, time="week", value="sales", levels=["store"])

    def test_empty_level_value_is_refused(self, tmp_path):
        path = _write(tmp_path, "store,week,sales\n,2024-01-01,10\n")
        with pytest.raises(DataError, match="empty value"):
            read_csv(path, time="week", value="sales", levels=["store"])

    def test_empty_file_is_refused(self, tmp_path):
        path = _write(tmp_path, "")
        with pytest.raises(DataError, match="empty file"):
            read_csv(path, time="week", value="sales", levels=["store"])

    def test_periods_sort_numerically_when_every_one_is_a_number(self, tmp_path):
        path = _write(tmp_path, "store,week,sales\nlahore,9,10\nlahore,10,20\nlahore,2,30\n")
        ds = read_csv(path, time="week", value="sales", levels=["store"])
        assert ds.periods == ["2", "9", "10"]

    def test_utf8_bom_does_not_break_the_first_column_name(self, tmp_path):
        path = tmp_path / "sales.csv"
        path.write_bytes(b"\xef\xbb\xbfstore,week,sales\nlahore,2024-01-01,10\n")
        ds = read_csv(path, time="week", value="sales", levels=["store"])
        assert ds.rows == 1
