"""Rolling-origin backtesting and scale-free accuracy.

Two things this file exists to get right.

**Leakage.** A backtest that lets the model see data after the forecast origin reports
an accuracy the system will never achieve in production, and it reports it
confidently. The split here is explicit: the forecaster is handed a *copy* of the
history up to the origin and nothing else, and a test watches what it was given.

**The verdict.** Two different yardsticks, which are easy to confuse:

* **MASE** divides the model's out-of-sample error by the *in-sample, one-step*
  error of (seasonal) naive on the training data. It is scale-free and comparable
  across series. It is **not** a head-to-head: a 4-step-ahead forecast is naturally
  worse than a 1-step one, so a perfectly sensible model, and seasonal naive itself,
  usually scores above 1.0 out of sample. "MASE > 1" does not mean "worse than doing
  nothing".
* **Relative MAE** (`relative_mae`, `beats_benchmark`) is the head-to-head. On the
  same folds, same origins and same horizon, the benchmark forecaster (seasonal naive
  with the backtest's `period`, or plain naive when `period == 1`) is run out of
  sample, and the model's MAE is divided by the benchmark's. Below 1.0 means the
  model beat the benchmark on exactly the forecasts it would have had to make.

MAPE is not offered: it divides by the actual, so the zeros intermittent demand is
full of make it infinite.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .models import check_horizon, clean_series, naive, naive_seasonal

Forecaster = Callable[[Sequence[float], int], list[float]]


class BacktestError(ValueError):
    """The backtest cannot be run as asked (too short a series, bad settings)."""


def _pairs(actual: Sequence[float], predicted: Sequence[float]) -> list[tuple[float, float]]:
    if len(actual) != len(predicted):
        raise ValueError(f"actual has {len(actual)} values but predicted has {len(predicted)}")
    return list(zip(actual, predicted, strict=True))


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    pairs = _pairs(actual, predicted)
    if not pairs:
        return 0.0
    return statistics.fmean(abs(a - p) for a, p in pairs)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    pairs = _pairs(actual, predicted)
    if not pairs:
        return 0.0
    return statistics.fmean((a - p) ** 2 for a, p in pairs) ** 0.5


def smape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Symmetric MAPE as a fraction, bounded at 2.0 (200%). Zero against zero counts as
    zero error rather than a division by zero."""
    pairs = _pairs(actual, predicted)
    if not pairs:
        return 0.0
    terms = []
    for a, p in pairs:
        denominator = (abs(a) + abs(p)) / 2
        terms.append(0.0 if denominator == 0 else abs(a - p) / denominator)
    return round(statistics.fmean(terms), 6)


def naive_scale(train: Sequence[float], *, period: int = 1) -> float:
    """Mean absolute error of one-step (seasonal) naive on the training data.

    This is the denominator of MASE, and it is computed **in-sample**: using the test
    period would let the difficulty of the test set flatter the model. Returns 0.0
    when the training data is not longer than one period.
    """
    if len(train) <= period:
        return 0.0
    errors = [abs(train[i] - train[i - period]) for i in range(period, len(train))]
    return statistics.fmean(errors)


def mase(
    actual: Sequence[float],
    predicted: Sequence[float],
    train: Sequence[float],
    *,
    period: int = 1,
) -> float:
    """Mean absolute scaled error. Returns NaN (not 0.0) when the training series has no
    naive error to scale by, i.e. it is flat or shorter than `period + 1` values;
    reporting 0.0 there would claim perfection. `summary()`/`to_dict()` turn that NaN
    into JSON `null`."""
    scale = naive_scale(train, period=period)
    if scale == 0:
        return float("nan")
    return round(mae(actual, predicted) / scale, 6)


def _json_float(value: float | None) -> float | None:
    if value is None or math.isnan(value):
        return None
    return value


@dataclass
class FoldResult:
    origin: int
    actual: list[float]
    predicted: list[float]
    mae: float
    rmse: float
    smape: float
    mase: float  # NaN when undefined; see `mase`
    benchmark_mae: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mase"] = _json_float(self.mase)
        return d


@dataclass
class BacktestResult:
    folds: list[FoldResult] = field(default_factory=list)
    horizon: int = 0
    initial: int = 0
    period: int = 1
    benchmark: str = "naive"

    def summary(self) -> dict[str, Any]:
        """Averages over folds. JSON-safe: undefined values are None, never NaN."""
        out: dict[str, Any] = {
            "folds": len(self.folds),
            "horizon": self.horizon,
            "initial": self.initial,
            "period": self.period,
            "benchmark": self.benchmark,
        }
        if not self.folds:
            return out
        model_mae = statistics.fmean(f.mae for f in self.folds)
        bench_mae = statistics.fmean(f.benchmark_mae for f in self.folds)
        mases = [f.mase for f in self.folds if not math.isnan(f.mase)]
        relative = round(model_mae / bench_mae, 6) if bench_mae > 0 else None
        out.update(
            {
                "mae": round(model_mae, 6),
                "rmse": round(statistics.fmean(f.rmse for f in self.folds), 6),
                "smape": round(statistics.fmean(f.smape for f in self.folds), 6),
                "mase": round(statistics.fmean(mases), 6) if mases else None,
                "mase_folds": len(mases),
                "benchmark_mae": round(bench_mae, 6),
                # The head-to-head: model MAE / benchmark MAE on the same folds.
                "relative_mae": relative,
                "beats_benchmark": (relative < 1.0) if relative is not None else None,
            }
        )
        return out

    def to_dict(self) -> dict[str, Any]:
        """Summary plus every fold, JSON-safe."""
        return {"summary": self.summary(), "folds": [f.to_dict() for f in self.folds]}


def _default_initial(horizon: int, period: int) -> int:
    # One full season plus one point, so both the seasonal-naive benchmark and the MASE
    # scale are defined at the first origin; at least two horizons and 8 points.
    return max(period + 1, 2 * horizon, 8)


def _check_int(name: str, value: int, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise BacktestError(f"{name} must be an int >= {low}, got {value!r}")
    return value


def rolling_origin(
    series: Sequence[float],
    forecaster: Forecaster,
    *,
    horizon: int = 4,
    initial: int | None = None,
    step: int = 1,
    period: int = 1,
    benchmark: Forecaster | None = None,
) -> BacktestResult:
    """Expanding-window backtest.

    The origin starts at `initial` (default `max(period + 1, 2 * horizon, 8)`) and
    advances by `step`; at each origin the forecaster sees only what precedes it and
    forecasts `horizon` values. Expanding rather than sliding because that is what
    production does. The benchmark (default: seasonal naive with `period`, or naive)
    is run on the same folds for the `relative_mae` verdict.

    Raises BacktestError if the series is too short for even one fold, instead of
    returning an empty result.
    """
    check_horizon(horizon)
    _check_int("step", step, 1)
    _check_int("period", period, 1)
    series = clean_series(series, name="series")
    if initial is None:
        initial = _default_initial(horizon, period)
    _check_int("initial", initial, 1)
    if initial + horizon > len(series):
        raise BacktestError(
            f"series has {len(series)} points but one fold needs initial + horizon = "
            f"{initial} + {horizon} = {initial + horizon}. Use a shorter horizon or pass "
            f"a smaller initial= (default is max(period + 1, 2 * horizon, 8))."
        )

    if benchmark is None:
        if period > 1:
            name = f"seasonal_naive(period={period})"

            def benchmark(h: Sequence[float], n: int) -> list[float]:
                return naive_seasonal(h, n, period=period)
        else:
            name = "naive"
            benchmark = naive
    else:
        name = getattr(benchmark, "__name__", "custom")

    result = BacktestResult(horizon=horizon, initial=initial, period=period, benchmark=name)
    for origin in range(initial, len(series) - horizon + 1, step):
        train = series[:origin]
        actual = series[origin : origin + horizon]
        # A copy, so a forecaster that mutates its input cannot corrupt later folds.
        predicted = _run(forecaster, train, horizon, "forecaster")
        reference = _run(benchmark, train, horizon, "benchmark")
        result.folds.append(
            FoldResult(
                origin=origin,
                actual=actual,
                predicted=predicted,
                mae=round(mae(actual, predicted), 6),
                rmse=round(rmse(actual, predicted), 6),
                smape=smape(actual, predicted),
                mase=mase(actual, predicted, train, period=period),
                benchmark_mae=round(mae(actual, reference), 6),
            )
        )
    return result


def _run(fn: Forecaster, train: list[float], horizon: int, label: str) -> list[float]:
    out = fn(list(train), horizon)
    try:
        values = [float(v) for v in out]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must return a list of numbers: {exc}") from exc
    if len(values) != horizon:
        raise ValueError(f"{label} returned {len(values)} values for horizon {horizon}")
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{label} returned a non-finite value: {values}")
    return values


def compare(
    series: Sequence[float], forecasters: dict[str, Forecaster], **kw: Any
) -> dict[str, dict[str, Any]]:
    """Score several forecasters on the same folds and the same benchmark.

    Same series, same origins, same horizon; otherwise the comparison measures the
    split rather than the models. Keyword arguments go to `rolling_origin`.
    """
    if not forecasters:
        raise ValueError("compare needs at least one forecaster")
    return {name: rolling_origin(series, fn, **kw).summary() for name, fn in forecasters.items()}


def backtest_reconciliation(
    hierarchy: Any,
    history: dict[str, Sequence[float]],
    forecaster: Forecaster,
    *,
    methods: Sequence[str] = ("bottom_up", "top_down", "mint_ols", "mint_wls", "weighted_blend"),
    horizon: int = 4,
    initial: int | None = None,
    step: int = 1,
    period: int = 1,
) -> dict[str, Any]:
    """Does reconciliation help? Rolling-origin MAE per node for the unreconciled base
    forecasts and for each reconciliation method, on the same folds.

    At each origin every node is forecast from its own history up to the origin, the
    base set is reconciled with each method (top-down proportions come from the same
    truncated history, so nothing leaks), and every node's error is recorded.
    `history` must hold every node and be coherent (use `Hierarchy.aggregate` or
    `read_csv`, which build it that way).
    """
    from .hierarchy import reconcile

    hierarchy._require_root()
    check_horizon(horizon)
    _check_int("step", step, 1)
    _check_int("period", period, 1)
    missing = [n for n in hierarchy.nodes if n not in history]
    if missing:
        raise BacktestError(f"no history for node {missing[0]!r}")
    series = {n: clean_series(history[n], name=f"history[{n!r}]") for n in hierarchy.nodes}
    lengths = {len(v) for v in series.values()}
    if len(lengths) != 1:
        raise BacktestError(f"every node needs the same history length; got {sorted(lengths)}")
    if not hierarchy.is_coherent(series):
        raise BacktestError("history is not coherent: every parent must equal its children's sum")
    n = lengths.pop()
    if initial is None:
        initial = _default_initial(horizon, period)
    _check_int("initial", initial, 1)
    if initial + horizon > n:
        raise BacktestError(
            f"history has {n} points but one fold needs initial + horizon = {initial + horizon}"
        )

    names = ["base", *methods]
    errors = {m: {node: 0.0 for node in hierarchy.nodes} for m in names}
    folds = 0
    for origin in range(initial, n - horizon + 1, step):
        train = {node: s[:origin] for node, s in series.items()}
        base = {node: _run(forecaster, train[node], horizon, "forecaster") for node in train}
        forecasts = {"base": base}
        for m in methods:
            forecasts[m] = reconcile(hierarchy, base, m, history=train)
        for m, fc in forecasts.items():
            for node, s in series.items():
                errors[m][node] += mae(s[origin : origin + horizon], fc[node])
        folds += 1

    levels = hierarchy.levels()
    out: dict[str, Any] = {"folds": folds, "horizon": horizon, "initial": initial, "methods": {}}
    for m in names:
        per_node = {node: round(v / folds, 6) for node, v in errors[m].items()}
        out["methods"][m] = {
            "mae_by_level": [
                round(statistics.fmean(per_node[x] for x in level), 6) for level in levels
            ],
            "mae_by_node": per_node,
        }
    out["levels"] = levels
    return out
