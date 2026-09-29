"""Command line: backtest or forecast a long-format sales CSV.

    python -m forecast backtest sales.csv --time week --value sales --period 52
    python -m forecast forecast sales.csv --time week --value sales \\
        --levels region,store --horizon 4 --period 52 --json

Errors go to stderr as one line and exit with status 2; `--json` output is valid JSON
(undefined numbers are null, never NaN).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from . import __version__
from .backtest import BacktestError, Forecaster, backtest_reconciliation, rolling_origin
from .hierarchy import METHODS, HierarchyError, reconcile
from .io import DataError, read_csv
from .models import (
    croston,
    drift,
    ets,
    holt,
    holt_winters,
    moving_average,
    naive,
    naive_seasonal,
    ses,
)

# name -> (needs a seasonal period?, factory taking the period)
MODELS: dict[str, tuple[bool, Callable[[int], Forecaster]]] = {
    "naive": (False, lambda p: naive),
    "seasonal_naive": (True, lambda p: partial(naive_seasonal, period=p)),
    "moving_average": (False, lambda p: moving_average),
    "drift": (False, lambda p: drift),
    "croston": (False, lambda p: croston),
    "ses": (False, lambda p: ses),
    "holt": (False, lambda p: holt),
    "holt_damped": (False, lambda p: partial(holt, damped=True)),
    "holt_winters": (True, lambda p: partial(holt_winters, period=p)),
    "ets": (False, lambda p: partial(ets, period=p)),
}
DEFAULT_MODELS = "naive,seasonal_naive,moving_average,ses,holt_damped,holt_winters,croston"


def _model_names(spec: str, period: int) -> list[str]:
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise ValueError(f"unknown model {unknown[0]!r}; choose from {', '.join(MODELS)}")
    if period == 1:
        # Without a season these are exactly naive / holt; listing them twice misleads.
        dropped = [n for n in names if MODELS[n][0]]
        names = [n for n in names if not MODELS[n][0]] or ["naive"]
        if dropped and spec != DEFAULT_MODELS:
            print(f"note: {', '.join(dropped)} need --period > 1; skipped", file=sys.stderr)
    return names


def _load(args: argparse.Namespace):
    levels = [c.strip() for c in args.levels.split(",") if c.strip()] if args.levels else []
    return read_csv(
        args.file,
        time=args.time,
        value=args.value,
        levels=levels,
        missing=args.missing,
        delimiter=args.delimiter,
    )


def _data_note(ds) -> dict[str, Any]:
    return {
        "rows": ds.rows,
        "periods": len(ds.periods),
        "first_period": ds.periods[0],
        "last_period": ds.periods[-1],
        "nodes": len(ds.hierarchy.nodes),
        "leaves": len(ds.hierarchy.leaves()),
        "filled_with_zero": ds.filled,
        "duplicate_rows_summed": ds.duplicates,
    }


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, list):
        return [_round(v) for v in x]
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return x


def cmd_backtest(args: argparse.Namespace) -> dict[str, Any]:
    ds = _load(args)
    names = _model_names(args.models, args.period)
    if args.node == "all":
        nodes = list(ds.hierarchy.nodes)
    else:
        if args.node not in ds.hierarchy.nodes:
            raise ValueError(f"no node {args.node!r}; nodes are {', '.join(ds.hierarchy.nodes)}")
        nodes = [args.node]
    results: dict[str, dict[str, Any]] = {}
    for node in nodes:
        per_model: dict[str, Any] = {}
        for name in names:
            fn = MODELS[name][1](args.period)
            try:
                per_model[name] = rolling_origin(
                    ds.history[node],
                    fn,
                    horizon=args.horizon,
                    period=args.period,
                    initial=args.initial,
                    step=args.step,
                ).summary()
            except BacktestError:
                raise
            except ValueError as exc:  # e.g. croston on a series with returns
                per_model[name] = {"error": str(exc)}
        results[node] = per_model
    return {"command": "backtest", "data": _data_note(ds), "results": results}


def cmd_forecast(args: argparse.Namespace) -> dict[str, Any]:
    ds = _load(args)
    if args.model not in MODELS:
        raise ValueError(f"unknown model {args.model!r}; choose from {', '.join(MODELS)}")
    fn = MODELS[args.model][1](args.period)
    h = ds.hierarchy
    base = {node: fn(ds.history[node], args.horizon) for node in h.nodes}
    if len(h.nodes) == 1:
        reconciled, method = base, "none (single series)"
    else:
        reconciled = reconcile(h, base, args.method, history=ds.history)
        method = args.method
    return {
        "command": "forecast",
        "data": _data_note(ds),
        "model": args.model,
        "method": method,
        "horizon": args.horizon,
        "base_coherent": h.is_coherent(base),
        "coherent": h.is_coherent(reconciled),
        "forecast": _round(reconciled),
        "base": _round(base),
    }


def cmd_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    ds = _load(args)
    if len(ds.hierarchy.nodes) == 1:
        raise ValueError("evaluate needs a hierarchy; pass --levels")
    if args.model not in MODELS:
        raise ValueError(f"unknown model {args.model!r}; choose from {', '.join(MODELS)}")
    result = backtest_reconciliation(
        ds.hierarchy,
        ds.history,
        MODELS[args.model][1](args.period),
        horizon=args.horizon,
        initial=args.initial,
        step=args.step,
        period=args.period,
    )
    levels = [c.strip() for c in args.levels.split(",") if c.strip()]
    return {
        "command": "evaluate",
        "data": _data_note(ds),
        "model": args.model,
        "level_names": [ds.hierarchy.root, *levels],
        **result,
    }


def _print_evaluate(out: dict[str, Any]) -> None:
    _print_data(out["data"])
    print(
        f"{out['folds']} folds, horizon {out['horizon']}, first origin {out['initial']}, "
        f"base model {out['model']}; mean MAE per node at each level"
    )
    names = out["level_names"]
    print()
    print(f"  {'method':15} " + " ".join(f"{n:>10}" for n in names))
    ranked = sorted(out["methods"].items(), key=lambda kv: kv[1]["mae_by_level"][0])
    for method, r in ranked:
        print(f"  {method:15} " + " ".join(f"{v:>10.3f}" for v in r["mae_by_level"]))
    print()
    print("'base' is the unreconciled forecast. Lower is better.")


def _print_data(d: dict[str, Any]) -> None:
    print(
        f"{d['rows']} rows, {d['periods']} periods ({d['first_period']} .. {d['last_period']}), "
        f"{d['nodes']} nodes; {d['filled_with_zero']} missing cells filled with 0"
    )


def _print_backtest(out: dict[str, Any]) -> None:
    _print_data(out["data"])
    for node, per_model in out["results"].items():
        first = next((s for s in per_model.values() if "error" not in s), None)
        print()
        if first:
            print(
                f"{node}: {first['folds']} folds, horizon {first['horizon']}, "
                f"benchmark {first['benchmark']}"
            )
        else:
            print(f"{node}:")
        print(f"  {'model':16} {'MAE':>10} {'rel MAE':>8} {'MASE':>7}  beats benchmark?")
        ranked = sorted(
            per_model.items(),
            key=lambda kv: (kv[1].get("relative_mae") is None, kv[1].get("relative_mae") or 0),
        )
        for name, s in ranked:
            if "error" in s:
                print(f"  {name:16} error: {s['error']}")
                continue
            rel = "-" if s["relative_mae"] is None else f"{s['relative_mae']:.3f}"
            mase = "-" if s["mase"] is None else f"{s['mase']:.3f}"
            verdict = (
                "-" if s["beats_benchmark"] is None else ("yes" if s["beats_benchmark"] else "no")
            )
            print(f"  {name:16} {s['mae']:>10.3f} {rel:>8} {mase:>7}  {verdict}")
    print()
    print("rel MAE = model MAE / benchmark MAE on the same folds (< 1 beats it).")
    print("MASE scales by in-sample one-step naive error, so > 1 is normal at horizon > 1.")


def _print_forecast(out: dict[str, Any]) -> None:
    _print_data(out["data"])
    print(
        f"model {out['model']}, reconciled with {out['method']}; "
        f"base coherent: {out['base_coherent']}, reconciled coherent: {out['coherent']}"
    )
    print()
    width = max(len(n) for n in out["forecast"])
    steps = len(next(iter(out["forecast"].values())))
    print(f"  {'node':{width}} " + " ".join(f"{'h+' + str(i + 1):>10}" for i in range(steps)))
    for node, values in out["forecast"].items():
        print(f"  {node:{width}} " + " ".join(f"{v:>10.2f}" for v in values))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m forecast", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("file", help="long-format CSV, one row per (levels..., period)")
        sp.add_argument("--time", required=True, help="period column (sorted; ISO dates work)")
        sp.add_argument("--value", required=True, help="numeric column to forecast")
        sp.add_argument("--levels", default="", help="comma-separated hierarchy columns, top first")
        sp.add_argument("--horizon", type=int, default=4, help="steps ahead (default 4)")
        sp.add_argument("--period", type=int, default=1, help="season length, e.g. 52 weekly")
        sp.add_argument(
            "--missing",
            choices=["zero", "error"],
            default="zero",
            help="leaf with no row for a period: 0 (default) or fail",
        )
        sp.add_argument("--delimiter", default=",")
        sp.add_argument("--json", action="store_true", help="print JSON instead of a table")

    bt = sub.add_parser("backtest", help="rolling-origin backtest against (seasonal) naive")
    common(bt)
    bt.add_argument(
        "--models", default=DEFAULT_MODELS, help=f"comma list from: {', '.join(MODELS)}"
    )
    bt.add_argument("--node", default="total", help="node to backtest, or 'all' (default total)")
    bt.add_argument("--initial", type=int, default=None, help="first origin (default auto)")
    bt.add_argument("--step", type=int, default=1, help="origin step (default 1)")

    fc = sub.add_parser("forecast", help="forecast every node and reconcile")
    common(fc)
    fc.add_argument("--model", default="ets", help=f"one of: {', '.join(MODELS)} (default ets)")
    fc.add_argument(
        "--method",
        default="mint_wls",
        choices=METHODS,
        help="reconciliation method (default mint_wls)",
    )
    ev = sub.add_parser("evaluate", help="backtest every reconciliation method")
    common(ev)
    ev.add_argument("--model", default="ets", help=f"base model (default ets): {', '.join(MODELS)}")
    ev.add_argument("--initial", type=int, default=None, help="first origin (default auto)")
    ev.add_argument("--step", type=int, default=1, help="origin step (default 1)")
    return p


COMMANDS = {"backtest": cmd_backtest, "forecast": cmd_forecast, "evaluate": cmd_evaluate}
PRINTERS = {"backtest": _print_backtest, "forecast": _print_forecast, "evaluate": _print_evaluate}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out = COMMANDS[args.command](args)
    except (DataError, HierarchyError, BacktestError, ValueError, TypeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(out, indent=2, allow_nan=False))
    else:
        PRINTERS[args.command](out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
