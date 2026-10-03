"""Does anything beat seasonal naive? Print `compare()` on the example total.

    python examples/compare_models.py

Run from the repo root (or with the package installed). The printed dict is what the
README's "Does anything beat seasonal naive?" section shows.
"""

import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecast.backtest import compare  # noqa: E402
from forecast.io import read_csv  # noqa: E402
from forecast.models import croston, holt_winters, naive, naive_seasonal  # noqa: E402

ds = read_csv(
    Path(__file__).with_name("weekly_sales.csv"), time="week", value="sales", levels=["region"]
)
series = ds.history["total"]
result = compare(
    series,
    {
        "seasonal": partial(naive_seasonal, period=52),
        "holt_winters": partial(holt_winters, period=52),
        "naive": naive,
        "croston": croston,
    },
    horizon=4,
    period=52,
)
keep = ("mase", "relative_mae", "beats_benchmark")

def _fields(summary: dict) -> dict:
    return {f: round(summary[f], 3) if isinstance(summary[f], float) else summary[f] for f in keep}


rows = [f' "{name}": {json.dumps(_fields(v))}' for name, v in result.items()]
print("{\n" + ",\n".join(rows) + "\n}")
