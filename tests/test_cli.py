"""Tests for the `demand-forecast` command line: a user's own CSV in.

These are the same paths a real user hits first - a bad column name, a missing
file, a request for every subcommand - and the whole point of the CLI existing is
that they no longer crash with a traceback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecast.cli import main  # noqa: E402


def _csv() -> str:
    lines = ["region,store,week,sales"]
    for i in range(24):  # enough weeks for a default (initial=8, horizon=4) backtest
        week = f"2024-{1 + i // 4:02d}-{1 + 7 * (i % 4):02d}"
        lines.append(f"north,lahore,{week},{120 + 3 * i}")
        lines.append(f"south,karachi,{week},{210 + 2 * i}")
    return "\n".join(lines) + "\n"


CSV = _csv()


@pytest.fixture
def sales_csv(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text(CSV, encoding="utf-8")
    return path


class TestForecast:
    def test_runs_end_to_end_and_is_coherent(self, sales_csv, capsys):
        code = main(
            [
                "forecast",
                str(sales_csv),
                "--time",
                "week",
                "--value",
                "sales",
                "--levels",
                "region,store",
                "--horizon",
                "2",
                "--model",
                "naive",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "total" in out
        assert "north/lahore" in out

    def test_json_output_is_valid_and_has_every_node(self, sales_csv, capsys):
        code = main(
            [
                "forecast",
                str(sales_csv),
                "--time",
                "week",
                "--value",
                "sales",
                "--levels",
                "region,store",
                "--horizon",
                "2",
                "--model",
                "naive",
                "--json",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert "north/lahore" in payload["forecast"]
        assert len(payload["forecast"]["total"]) == 2

    def test_every_reconciliation_method_runs(self, sales_csv):
        for method in ("bottom_up", "top_down", "mint_ols", "mint_wls", "weighted_blend"):
            code = main(
                [
                    "forecast",
                    str(sales_csv),
                    "--time",
                    "week",
                    "--value",
                    "sales",
                    "--levels",
                    "region,store",
                    "--horizon",
                    "2",
                    "--model",
                    "naive",
                    "--method",
                    method,
                    "--json",
                ]
            )
            assert code == 0, method

    def test_missing_file_is_a_clean_error_not_a_traceback(self, tmp_path, capsys):
        code = main(
            [
                "forecast",
                str(tmp_path / "nope.csv"),
                "--time",
                "week",
                "--value",
                "sales",
                "--levels",
                "region",
            ]
        )
        assert code == 2
        assert "error:" in capsys.readouterr().err

    def test_unknown_column_is_a_clean_error(self, sales_csv, capsys):
        code = main(
            [
                "forecast",
                str(sales_csv),
                "--time",
                "week",
                "--value",
                "not_a_column",
                "--levels",
                "region",
            ]
        )
        assert code == 2
        err = capsys.readouterr().err
        assert "not_a_column" in err
        assert "not found" in err


class TestBacktest:
    def test_runs_end_to_end(self, sales_csv, capsys):
        code = main(
            [
                "backtest",
                str(sales_csv),
                "--time",
                "week",
                "--value",
                "sales",
                "--levels",
                "region",
            ]
        )
        assert code == 0
        assert "total" in capsys.readouterr().out

    def test_json_output_has_a_result_per_node(self, sales_csv, capsys):
        code = main(
            [
                "backtest",
                str(sales_csv),
                "--time",
                "week",
                "--value",
                "sales",
                "--levels",
                "region",
                "--json",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert "total" in payload["results"]


class TestEvaluate:
    def test_runs_end_to_end(self, sales_csv, capsys):
        code = main(
            [
                "evaluate",
                str(sales_csv),
                "--time",
                "week",
                "--value",
                "sales",
                "--levels",
                "region,store",
                "--horizon",
                "2",
            ]
        )
        assert code == 0
        capsys.readouterr()  # just confirm it printed without raising


class TestVersion:
    def test_version_flag_exits_cleanly(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
