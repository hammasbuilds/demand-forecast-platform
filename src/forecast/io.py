"""Read a long-format sales table into a hierarchy and one series per node.

The shape almost every sales export has: one row per (location..., period) with a
number. For example

    region,store,week,sales
    north,lahore,2024-01-01,120
    north,islamabad,2024-01-01,80
    south,karachi,2024-01-01,210

`read_csv(path, time="week", value="sales", levels=["region", "store"])` builds the
tree total -> north -> north/lahore, ... and a history for every node, internal
nodes being the sum of their leaves. Standard library `csv` only.

Decisions that change numbers, all explicit:

* Periods are sorted numerically when every one parses as a number, otherwise as
  text, which is chronological for ISO dates (2024-01-08) but not for 1/8/2024.
* Several rows for the same leaf and period (transaction-level data) are summed.
* A leaf with no row for a period is 0 by default (`missing="zero"`), because many
  exports omit zero-sales rows; `missing="error"` raises instead. The count of filled
  cells is reported so a gap is never invisible.
* An empty, non-numeric, NaN or infinite value raises with its line number.
"""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

from .hierarchy import Hierarchy


class DataError(ValueError):
    pass


@dataclass
class Dataset:
    hierarchy: Hierarchy
    history: dict[str, list[float]]  # every node, oldest period first
    periods: list[str]
    rows: int
    filled: int  # (leaf, period) cells with no row, filled with 0
    duplicates: int  # rows summed into a (leaf, period) cell that already had one


def _sort_periods(periods: set[str]) -> list[str]:
    try:
        return sorted(periods, key=float)
    except ValueError:
        return sorted(periods)


def read_csv(
    path: str | os.PathLike[str],
    *,
    time: str,
    value: str,
    levels: Sequence[str] = (),
    missing: str = "zero",
    root: str = "total",
    sep: str = "/",
    delimiter: str = ",",
) -> Dataset:
    """Load a long-format CSV. See the module docstring for the rules."""
    if missing not in ("zero", "error"):
        raise ValueError(f"missing must be 'zero' or 'error', got {missing!r}")
    levels = list(levels)
    # utf-8-sig: Excel on Windows writes a BOM that would otherwise glue itself to the
    # first column name and make it "not found".
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        header = [c.strip() for c in (reader.fieldnames or [])]
        if not header:
            raise DataError(f"{path}: empty file or no header row")
        reader.fieldnames = header
        needed = [time, value, *levels]
        absent = [c for c in needed if c not in header]
        if absent:
            raise DataError(
                f"{path}: column {absent[0]!r} not found; columns are {', '.join(header)}"
            )

        cells: dict[tuple[tuple[str, ...], str], float] = {}
        duplicates = 0
        rows = 0
        for line, row in enumerate(reader, start=2):
            rows += 1
            raw = (row.get(value) or "").strip()
            try:
                number = float(raw)
            except ValueError:
                raise DataError(f"{path}:{line}: {value}={raw!r} is not a number") from None
            if not math.isfinite(number):
                raise DataError(f"{path}:{line}: {value}={raw!r}; fill missing values first")
            period = (row.get(time) or "").strip()
            if not period:
                raise DataError(f"{path}:{line}: empty {time}")
            key_path = tuple((row.get(c) or "").strip() for c in levels)
            if any(not p for p in key_path):
                raise DataError(f"{path}:{line}: empty value in one of {', '.join(levels)}")
            key = (key_path, period)
            if key in cells:
                duplicates += 1
                cells[key] += number
            else:
                cells[key] = number

    if not rows:
        raise DataError(f"{path}: header but no data rows")

    periods = _sort_periods({p for _, p in cells})
    paths = sorted({k for k, _ in cells})
    hierarchy = Hierarchy.from_paths(paths if levels else [], root=root, sep=sep)
    leaf_name = {p: (sep.join(p) if p else root) for p in paths}

    leaf_history: dict[str, list[float]] = {}
    filled = 0
    for p in paths:
        series = []
        for period in periods:
            v = cells.get((p, period))
            if v is None:
                if missing == "error":
                    raise DataError(f"{path}: no row for {leaf_name[p]!r} in period {period!r}")
                filled += 1
                v = 0.0
            series.append(v)
        leaf_history[leaf_name[p]] = series

    history = hierarchy.aggregate(leaf_history)
    return Dataset(
        hierarchy=hierarchy,
        history={k: list(v) for k, v in history.items()},
        periods=periods,
        rows=rows,
        filled=filled,
        duplicates=duplicates,
    )
