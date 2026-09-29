"""Write examples/weekly_sales.csv: three years of synthetic weekly sales.

    python examples/make_weekly_sales.py

Two regions, five stores, 156 weeks from 2023-01-02. Each store has its own level,
a yearly season (stronger in the south), a slow trend and noise; one store opens late
and has no rows before its first week, which the loader fills with zeros. Seeded, so
the file is identical on every run. Synthetic by design: the point is a file anyone
can regenerate and check the README numbers against, not a claim about real stores.
"""

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

STORES = {
    # (region, store): (base level, season amplitude, trend per week, noise sd)
    ("north", "lahore"): (120.0, 25.0, 0.15, 9.0),
    ("north", "islamabad"): (80.0, 15.0, 0.05, 7.0),
    ("north", "multan"): (45.0, 10.0, 0.20, 6.0),
    ("south", "karachi"): (210.0, 55.0, 0.10, 14.0),
    ("south", "hyderabad"): (60.0, 20.0, -0.05, 6.0),
}
LATE_OPENING = ("north", "multan")
OPENS_AT_WEEK = 20
WEEKS = 156


def main() -> None:
    rng = random.Random(42)
    start = date(2023, 1, 2)
    out = Path(__file__).with_name("weekly_sales.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["region", "store", "week", "sales"])
        for t in range(WEEKS):
            week = (start + timedelta(weeks=t)).isoformat()
            for (region, store), (level, amp, trend, sd) in STORES.items():
                if (region, store) == LATE_OPENING and t < OPENS_AT_WEEK:
                    continue
                season = amp * math.sin(2 * math.pi * t / 52) + 0.4 * amp * math.cos(
                    4 * math.pi * t / 52
                )
                value = max(0.0, level + trend * t + season + rng.gauss(0, sd))
                w.writerow([region, store, week, round(value)])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
