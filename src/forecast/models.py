"""Forecasters.

Every forecaster has the same shape, `(history, horizon) -> list[float]`, so any of
them (or your own) can be dropped into the backtest. Keyword-only settings such as
`period` are bound with a lambda or `functools.partial`.

Two families live here:

**Baselines** (`naive`, `naive_seasonal`, `moving_average`, `drift`). A forecasting
project that opens with gradient boosting has skipped the question that decides
whether it is worth doing: *does anything beat seasonal naive?* In retail it often
does not, so the baselines are what everything else is scored against.

**Smoothing models** (`croston`, `ses`, `holt`, `holt_winters`, `ets`). Croston is
for intermittent demand, long runs of zeros with occasional orders, which ordinary
smoothing handles badly. The exponential-smoothing (ETS) family is the additive-error
subset: simple (level only), Holt (additive trend, optionally damped) and additive
Holt-Winters (trend plus one additive season). Parameters are chosen by a grid search
that minimises the in-sample one-step squared error. That is not the maximum
likelihood fit `statsmodels` would do, and there are no multiplicative components and
no prediction intervals; it is the honest stdlib version.

The smoothing models drop *leading* zeros (periods before a product or store
launched) before fitting; zeros after the first sale are kept.

Every forecaster rejects NaN, infinities and non-numbers in the history instead of
returning a NaN forecast, and rejects a horizon below 1.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from numbers import Real

# Grids for the smoothing parameters. Coarse on purpose: the SSE surface is flat near
# its minimum, and a finer grid mostly fits noise on the short series demand has.
_ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_BETAS = (0.01, 0.05, 0.1, 0.2, 0.3)
_GAMMAS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
_PHIS = (0.8, 0.9, 0.95, 0.98, 1.0)


def clean_series(history: Sequence[float], *, name: str = "history") -> list[float]:
    """Return `history` as a list of floats, or raise ValueError naming the bad value.

    Silent NaN is the classic way a forecasting pipeline produces garbage: one missing
    week turns every downstream forecast and metric into NaN, and nothing says why.
    """
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError(f"{name} must be a sequence of numbers, got {type(history).__name__}")
    out: list[float] = []
    for i, v in enumerate(history):
        if isinstance(v, bool) or not isinstance(v, Real):
            raise TypeError(f"{name}[{i}] is {v!r}, not a number")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"{name}[{i}] is {v!r}; fill or drop missing values first")
        out.append(f)
    return out


def check_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError(f"horizon must be an int, got {horizon!r}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return horizon


def _check_period(period: int) -> int:
    if isinstance(period, bool) or not isinstance(period, int) or period < 1:
        raise ValueError(f"period must be an int >= 1, got {period!r}")
    return period


def _check_unit(name: str, value: float | None, *, allow_zero: bool = False) -> None:
    if value is None:
        return
    low_ok = value >= 0 if allow_zero else value > 0
    if not (isinstance(value, Real) and low_ok and value <= 1):
        bound = "0 <=" if allow_zero else "0 <"
        raise ValueError(f"{name} must satisfy {bound} {name} <= 1, got {value!r}")


# --- baselines ----------------------------------------------------------------------


def naive(history: Sequence[float], horizon: int) -> list[float]:
    """Last value, repeated. The floor any model must clear. Empty history forecasts 0."""
    check_horizon(horizon)
    history = clean_series(history)
    if not history:
        return [0.0] * horizon
    return [history[-1]] * horizon


def naive_seasonal(history: Sequence[float], horizon: int, *, period: int) -> list[float]:
    """Same period last season. In retail this is a genuinely strong competitor.

    With less than one full season of history it falls back to `naive`.
    """
    check_horizon(horizon)
    _check_period(period)
    history = clean_series(history)
    if len(history) < period:
        return naive(history, horizon)
    return [history[-period + (i % period)] for i in range(horizon)]


def moving_average(history: Sequence[float], horizon: int, *, window: int = 4) -> list[float]:
    """Mean of the last `window` values, repeated."""
    check_horizon(horizon)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(f"window must be an int >= 1, got {window!r}")
    history = clean_series(history)
    if not history:
        return [0.0] * horizon
    return [statistics.fmean(history[-window:])] * horizon


def drift(history: Sequence[float], horizon: int) -> list[float]:
    """Last value plus the average per-period change. A trend model with no fitting."""
    check_horizon(horizon)
    history = clean_series(history)
    if len(history) < 2:
        return naive(history, horizon)
    slope = (history[-1] - history[0]) / (len(history) - 1)
    return [history[-1] + slope * (i + 1) for i in range(horizon)]


# --- intermittent demand ------------------------------------------------------------


def croston(history: Sequence[float], horizon: int, *, alpha: float = 0.1) -> list[float]:
    """Croston's method for intermittent demand.

    Two exponentially smoothed series rather than one: the **size** of a non-zero
    demand, and the **interval** between non-zero demands. The forecast is size
    divided by interval, a demand *rate* per period.

    Smoothing the raw series lets the zeros drag the level toward zero between orders,
    so the forecast is too low right before an order and too high right after.
    Separating size from timing removes that oscillation.

    Demand cannot be negative here: a net-returns series is not intermittent demand,
    and silently dropping the negatives (as a `v > 0` filter would) biases the rate.
    """
    check_horizon(horizon)
    _check_unit("alpha", alpha)
    history = clean_series(history)
    negative = [i for i, v in enumerate(history) if v < 0]
    if negative:
        raise ValueError(
            f"croston needs non-negative demand; history[{negative[0]}] = "
            f"{history[negative[0]]} (use naive/ses for a series with returns)"
        )
    demands = [(i, v) for i, v in enumerate(history) if v > 0]
    if not demands:
        return [0.0] * horizon
    if len(demands) == 1:
        # One observation gives a size but no interval to estimate.
        return [demands[0][1]] * horizon

    size = demands[0][1]
    interval = float(demands[1][0] - demands[0][0])

    previous_index = demands[0][0]
    for index, value in demands[1:]:
        gap = index - previous_index
        size += alpha * (value - size)
        interval += alpha * (gap - interval)
        previous_index = index

    return [size / interval] * horizon


# --- exponential smoothing (ETS, additive errors) -----------------------------------


def _launched(y: list[float]) -> list[float]:
    """Drop leading zeros: periods before a product or store existed are not demand.

    Without this a store that opened in week 20 starts its level (and, for
    Holt-Winters, its seasonal profile) at zero and drags every forecast down.
    Zeros after the first non-zero value are kept; they are real.
    """
    for i, v in enumerate(y):
        if v != 0:
            return y[i:]
    return []


def _ses_sse(y: list[float], alpha: float) -> tuple[float, float]:
    level = y[0]
    sse = 0.0
    for v in y[1:]:
        err = v - level
        sse += err * err
        level += alpha * err
    return sse, level


def ses(history: Sequence[float], horizon: int, *, alpha: float | None = None) -> list[float]:
    """Simple exponential smoothing, ETS(A,N,N): a level and nothing else.

    `alpha=None` picks alpha from a grid by in-sample one-step squared error.
    """
    check_horizon(horizon)
    _check_unit("alpha", alpha)
    y = _launched(clean_series(history))
    if len(y) < 2:
        return naive(y, horizon)
    alphas = (alpha,) if alpha is not None else _ALPHAS
    _, level = min((_ses_sse(y, a) for a in alphas), key=lambda t: t[0])
    return [level] * horizon


def _holt_run(y: list[float], alpha: float, beta: float, phi: float) -> tuple[float, float, float]:
    level, trend = y[0], y[1] - y[0]
    sse = 0.0
    for v in y[1:]:
        fitted = level + phi * trend
        err = v - fitted
        sse += err * err
        new_level = fitted + alpha * err
        trend = phi * trend + beta * (new_level - level - phi * trend)
        level = new_level
    return sse, level, trend


def _damped_steps(phi: float, horizon: int) -> list[float]:
    out, acc, p = [], 0.0, 1.0
    for _ in range(horizon):
        p *= phi
        acc += p
        out.append(acc)
    return out


def holt(
    history: Sequence[float],
    horizon: int,
    *,
    alpha: float | None = None,
    beta: float | None = None,
    damped: bool = False,
    phi: float | None = None,
) -> list[float]:
    """Holt's linear trend, ETS(A,A,N), or ETS(A,Ad,N) with `damped=True`.

    An undamped trend extrapolates forever, which is how a two-good-weeks blip becomes
    a 12-week overstock; damping flattens the trend toward the horizon.
    Parameters left as None are chosen by grid search on in-sample one-step SSE.
    """
    check_horizon(horizon)
    _check_unit("alpha", alpha)
    _check_unit("beta", beta)
    _check_unit("phi", phi)
    y = _launched(clean_series(history))
    if len(y) < 3:
        return drift(y, horizon)
    alphas = (alpha,) if alpha is not None else _ALPHAS
    betas = (beta,) if beta is not None else _BETAS
    if phi is not None:
        phis: tuple[float, ...] = (phi,)
    else:
        phis = _PHIS if damped else (1.0,)
    best = None
    for a in alphas:
        for b in betas:
            for p in phis:
                sse, level, trend = _holt_run(y, a, b, p)
                if best is None or sse < best[0]:
                    best = (sse, level, trend, p)
    assert best is not None
    _, level, trend, p = best
    return [level + s * trend for s in _damped_steps(p, horizon)]


def _hw_run(
    y: list[float], period: int, alpha: float, beta: float, gamma: float
) -> tuple[float, float, float, list[float]]:
    # Classical initialisation from the first two seasons: level = mean of season 1,
    # trend = mean per-period change between the two season means, season = the first
    # season's deviations from its mean.
    first = y[:period]
    second = y[period : 2 * period]
    level = statistics.fmean(first)
    trend = (statistics.fmean(second) - level) / period
    season = [v - level for v in first]
    sse = 0.0
    for t in range(period, len(y)):
        s = season[t % period]
        fitted = level + trend + s
        err = y[t] - fitted
        sse += err * err
        new_level = level + trend + alpha * err
        trend = trend + beta * (new_level - level - trend)
        season[t % period] = s + gamma * (y[t] - new_level - s)
        level = new_level
    return sse, level, trend, season


def holt_winters(
    history: Sequence[float],
    horizon: int,
    *,
    period: int,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
) -> list[float]:
    """Additive Holt-Winters, ETS(A,A,A): level, trend and one additive season.

    Needs two full seasons of history to initialise; with less it falls back to
    `holt`. Parameters left as None are chosen by grid search on in-sample one-step
    SSE, so a long weekly series (period 52) with the full grid costs a few hundred
    passes over the data, well under a second.
    """
    check_horizon(horizon)
    _check_period(period)
    _check_unit("alpha", alpha)
    _check_unit("beta", beta)
    _check_unit("gamma", gamma, allow_zero=True)
    y = _launched(clean_series(history))
    if period == 1 or len(y) < 2 * period:
        return holt(y, horizon, alpha=alpha, beta=beta)
    alphas = (alpha,) if alpha is not None else _ALPHAS
    betas = (beta,) if beta is not None else _BETAS
    gammas = (gamma,) if gamma is not None else _GAMMAS
    best = None
    for a in alphas:
        for b in betas:
            for g in gammas:
                run = _hw_run(y, period, a, b, g)
                if best is None or run[0] < best[0]:
                    best = run
    assert best is not None
    _, level, trend, season = best
    n = len(y)
    return [level + (h + 1) * trend + season[(n + h) % period] for h in range(horizon)]


def ets(
    history: Sequence[float],
    horizon: int,
    *,
    period: int = 1,
    trend: bool | None = None,
    seasonal: bool | None = None,
    damped: bool = True,
) -> list[float]:
    """Pick an additive ETS model and forecast with it.

    With `trend`/`seasonal` left as None the choice is automatic: seasonal when
    `period > 1` and there are two full seasons of history; a (damped) trend when the
    holdout of the last `max(period, 4)` points says Holt beats simple smoothing on
    it. Pass True/False to force a component.
    """
    check_horizon(horizon)
    _check_period(period)
    y = _launched(clean_series(history))
    if seasonal is None:
        seasonal = period > 1 and len(y) >= 2 * period
    if seasonal:
        if period < 2:
            raise ValueError("seasonal=True needs period >= 2")
        return holt_winters(y, horizon, period=period)
    if trend is None:
        hold = max(period, 4)
        if len(y) < hold + 4:
            trend = False
        else:
            train, test = y[:-hold], y[-hold:]
            e_ses = sum(abs(a - p) for a, p in zip(test, ses(train, hold), strict=True))
            e_holt = sum(
                abs(a - p) for a, p in zip(test, holt(train, hold, damped=damped), strict=True)
            )
            trend = e_holt < e_ses
    if trend:
        return holt(y, horizon, damped=damped)
    return ses(y, horizon)
