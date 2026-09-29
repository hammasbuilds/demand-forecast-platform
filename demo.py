"""Forecast a store hierarchy, make the numbers add up, and check against seasonal naive.

    python demo.py

Three parts, all on files and code in this repo, no network:

1. examples/weekly_sales.csv (3 years, 2 regions, 5 stores): ETS forecasts of every
   node, 4 weeks ahead, do not add up; three reconciliation methods make them.
2. A rolling-origin backtest of the total: does anything beat seasonal naive, out of
   sample, on the same folds?
3. An intermittent spare-part series: Croston against the naive benchmark.
"""

import random
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from forecast import (  # noqa: E402
    backtest_reconciliation,
    compare,
    croston,
    drift,
    ets,
    holt,
    holt_winters,
    moving_average,
    naive,
    naive_seasonal,
    read_csv,
    reconcile,
    ses,
)

HORIZON = 4
PERIOD = 52

data = read_csv(
    ROOT / "examples" / "weekly_sales.csv", time="week", value="sales", levels=["region", "store"]
)
h = data.hierarchy

print("INPUT")
print(
    f"   examples/weekly_sales.csv: {data.rows} rows, {len(data.periods)} weeks "
    f"({data.periods[0]} .. {data.periods[-1]})"
)
print(f"   {len(h.nodes)} nodes: " + ", ".join(h.nodes))
print(f"   {data.filled} store-weeks with no row (north/multan opened late) filled with 0")
print()

# --- 1. reconciliation --------------------------------------------------------------
base = {node: ets(data.history[node], HORIZON, period=PERIOD) for node in h.nodes}

print("OUTPUT")
print(f"1. ETS forecast of every node, {HORIZON} weeks ahead")
print(f"   coherent as forecast?  {h.is_coherent(base)}")
for step in range(HORIZON):
    kids = base["north"][step] + base["south"][step]
    print(
        f"      week +{step + 1}: north + south = {kids:7.1f}, total says "
        f"{base['total'][step]:7.1f}  (off by {kids - base['total'][step]:+.1f})"
    )
print()
print(f"   {'method':15} {'coherent':>8}  total, weeks +1..+{HORIZON}")
for method in ("bottom_up", "mint_wls", "weighted_blend"):
    rec = reconcile(h, base, method)
    totals = "  ".join(f"{v:7.1f}" for v in rec["total"])
    print(f"   {method:15} {h.is_coherent(rec)!s:>8}  {totals}")
print()

# Coherent is not the same as accurate. Which method is closest to what happened?
ev = backtest_reconciliation(
    h,
    data.history,
    partial(ets, period=PERIOD),
    horizon=HORIZON,
    period=PERIOD,
    initial=2 * PERIOD,
    step=2,
)
print(
    f"   Which reconciliation is most accurate? {ev['folds']} rolling origins over year 3, "
    f"MAE per node, {HORIZON} weeks ahead"
)
print(f"   {'method':15} {'total':>7} {'regions':>8} {'stores':>7}")
ranked = sorted(ev["methods"].items(), key=lambda kv: kv[1]["mae_by_level"][0])
for method, r in ranked:
    top, mid, leaf = r["mae_by_level"]
    print(f"   {method:15} {top:7.2f} {mid:8.2f} {leaf:7.2f}")
best = ranked[0][0]
print(f"   On this data {best} is best at the total; 'base' is the unreconciled forecast.")
print()

# --- 2. does anything beat seasonal naive? ------------------------------------------
models = {
    "seasonal_naive": partial(naive_seasonal, period=PERIOD),
    "naive": naive,
    "ses": ses,
    "holt_damped": partial(holt, damped=True),
    "holt_winters": partial(holt_winters, period=PERIOD),
}
scores = compare(data.history["total"], models, horizon=HORIZON, period=PERIOD)
first = next(iter(scores.values()))
print(
    f"2. Backtest of the total: {first['folds']} rolling origins, {HORIZON} weeks ahead, "
    f"benchmark {first['benchmark']}"
)
print(f"   {'forecaster':15} {'MAE':>7} {'rel MAE':>8} {'MASE':>6}  beats seasonal naive?")
for name, s in sorted(scores.items(), key=lambda kv: kv[1]["relative_mae"]):
    print(
        f"   {name:15} {s['mae']:7.2f} {s['relative_mae']:8.3f} {s['mase']:6.3f}  "
        f"{'yes' if s['beats_benchmark'] else 'no'}"
    )
winners = [n for n, s in scores.items() if s["beats_benchmark"]]
print(
    "   rel MAE is MAE divided by seasonal naive's MAE on the same folds. "
    f"Beating it: {', '.join(winners) or 'nothing'}."
)
print()

# --- 3. intermittent demand ---------------------------------------------------------
rng = random.Random(3)
spare_part = [rng.choice([0, 0, 0, 0, 0, 2, 3]) for _ in range(40)]
nonzero = sum(1 for v in spare_part if v)
print(f"3. Intermittent series: {nonzero} non-zero days out of {len(spare_part)}")
print(f"      {spare_part}")
results = compare(
    spare_part,
    {"croston": croston, "moving_average": moving_average, "naive": naive, "drift": drift},
    horizon=HORIZON,
    initial=20,
)
first = next(iter(results.values()))
print(f"   {first['folds']} rolling origins, {HORIZON} days ahead, benchmark {first['benchmark']}")
print(f"   {'forecaster':15} {'MAE':>7} {'rel MAE':>8} {'MASE':>6}  beats naive?")
for name, s in sorted(results.items(), key=lambda kv: kv[1]["relative_mae"]):
    print(
        f"   {name:15} {s['mae']:7.3f} {s['relative_mae']:8.3f} {s['mase']:6.3f}  "
        f"{'yes' if s['beats_benchmark'] else 'no'}"
    )
c = results["croston"]
print(
    f"   Croston cuts naive's error by {100 * (1 - c['relative_mae']):.0f}%, yet its MASE is "
    f"{c['mase']:.2f}:"
)
print("   worse than one-step naive *in sample*, the yardstick MASE uses. On a series")
print("   this sparse, a rate forecast plus a service-level stock is the realistic plan.")
