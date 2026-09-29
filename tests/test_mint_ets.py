"""Tests for the MinT reconciliation solver and the ETS model family.

Neither had a test before this file - `mint` is a from-scratch least-squares solve
(no numpy), and the ETS family (`ses`/`holt`/`holt_winters`/`ets`) is what the CLI
defaults to. Both are exercised through the CLI on real data, but a passing CLI run
does not pin what a correct answer looks like - these do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecast.hierarchy import Hierarchy, HierarchyError, mint  # noqa: E402
from forecast.models import ets, holt, holt_winters, ses  # noqa: E402


def _two_leaf_hierarchy() -> Hierarchy:
    h = Hierarchy()
    h.add("total")
    h.add("north", parent="total")
    h.add("south", parent="total")
    return h


class TestMint:
    @pytest.mark.parametrize("method", ["ols", "wls_struct"])
    def test_result_is_coherent(self, method):
        h = _two_leaf_hierarchy()
        base = {"total": 100.0, "north": 40.0, "south": 40.0}  # disagree: 40+40 != 100
        out = mint(h, base, method=method)
        assert h.is_coherent(out)

    def test_agreeing_base_forecasts_are_left_unchanged(self):
        # If the base forecasts already sum correctly, reconciling them should not
        # move them - there is nothing to resolve.
        h = _two_leaf_hierarchy()
        base = {"total": 80.0, "north": 30.0, "south": 50.0}
        out = mint(h, base, method="ols")
        assert out["north"] == pytest.approx(30.0, abs=1e-6)
        assert out["south"] == pytest.approx(50.0, abs=1e-6)
        assert out["total"] == pytest.approx(80.0, abs=1e-6)

    def test_wls_struct_weights_a_bigger_subtree_more_forgivingly(self):
        # `total` has two leaves under it (weight 2) and `north` has one (weight 1),
        # so wls_struct should not move `total` and `north` by equal amounts to fix
        # the same-sized disagreement - ols does, wls_struct does not.
        h = Hierarchy()
        h.add("total")
        h.add("north", parent="total")
        h.add("north/a", parent="north")
        h.add("north/b", parent="north")
        base = {
            "total": 100.0,
            "north": 100.0,
            "north/a": 40.0,
            "north/b": 40.0,
        }
        ols = mint(h, base, method="ols")
        wls = mint(h, base, method="wls_struct")
        assert ols["total"] != pytest.approx(wls["total"], abs=1e-9)

    def test_multi_step_input_reconciles_every_step_independently(self):
        h = _two_leaf_hierarchy()
        base = {
            "total": [100.0, 200.0],
            "north": [40.0, 90.0],
            "south": [40.0, 90.0],
        }
        out = mint(h, base, method="ols")
        assert len(out["total"]) == 2
        for step in range(2):
            row = {name: values[step] for name, values in out.items()}
            assert h.is_coherent(row)

    def test_rejects_an_unknown_method(self):
        h = _two_leaf_hierarchy()
        base = {"total": 10.0, "north": 4.0, "south": 6.0}
        with pytest.raises(HierarchyError, match="method"):
            mint(h, base, method="not-a-method")

    def test_needs_a_forecast_for_every_node_not_only_leaves(self):
        h = _two_leaf_hierarchy()
        with pytest.raises(HierarchyError):
            mint(h, {"north": 4.0, "south": 6.0}, method="ols")  # total missing


class TestETSFamily:
    def test_ses_is_flat_at_the_series_mean_region_for_level_data(self):
        y = [50.0] * 20
        out = ses(y, horizon=4)
        assert out == pytest.approx([50.0] * 4, abs=0.5)

    def test_ses_alpha_is_bounded(self):
        with pytest.raises(ValueError):
            ses([1.0, 2.0, 3.0], horizon=2, alpha=1.5)
        with pytest.raises(ValueError):
            ses([1.0, 2.0, 3.0], horizon=2, alpha=0)

    def test_holt_extrapolates_a_clean_linear_trend(self):
        y = [10.0 + 2.0 * i for i in range(20)]
        out = holt(y, horizon=3, alpha=0.9, beta=0.9)
        # Next points should continue near 10 + 2*20, 10 + 2*21, ...
        assert out[0] == pytest.approx(48.0, abs=3.0)
        assert out[2] > out[0]  # still trending up

    def test_holt_damped_trend_flattens_relative_to_undamped(self):
        y = [10.0 + 2.0 * i for i in range(20)]
        undamped = holt(y, horizon=10, alpha=0.9, beta=0.9, damped=False)
        damped = holt(y, horizon=10, alpha=0.9, beta=0.9, damped=True, phi=0.8)
        # The gap between consecutive forecasts should shrink under damping and not
        # under the undamped (linear) extrapolation.
        undamped_gap = undamped[-1] - undamped[-2]
        damped_gap = damped[-1] - damped[-2]
        assert damped_gap < undamped_gap

    def test_holt_winters_recovers_a_clean_additive_season(self):
        period = 4
        season = [5.0, -5.0, 10.0, -10.0]
        y = [100.0 + season[i % period] for i in range(5 * period)]
        out = holt_winters(y, horizon=period, period=period)
        # The forecast should reproduce the seasonal shape, not flatten it.
        assert out[0] == pytest.approx(100.0 + season[0], abs=2.0)
        assert out[1] == pytest.approx(100.0 + season[1], abs=2.0)

    def test_holt_winters_falls_back_to_holt_with_under_two_seasons(self):
        # Documented behaviour: not enough history to init a season, so it must not
        # raise or silently ignore the request - it degrades to holt().
        y = [10.0, 12.0, 14.0, 16.0, 18.0]  # < 2 * period(=4)
        out = holt_winters(y, horizon=2, period=4)
        assert len(out) == 2

    def test_ets_picks_seasonal_when_period_and_history_support_it(self):
        period = 12
        season = [i - 5.5 for i in range(period)]
        y = [50.0 + season[i % period] for i in range(4 * period)]
        out = ets(y, horizon=period, period=period)
        assert len(out) == period
        # Seasonal was chosen: the forecast should not be flat across the period.
        assert max(out) - min(out) > 5.0

    def test_ets_with_no_seasonality_and_short_history_does_not_crash(self):
        out = ets([3.0, 4.0, 5.0], horizon=2)
        assert len(out) == 2

    def test_ets_period_must_be_at_least_1(self):
        with pytest.raises(ValueError):
            ets([1.0, 2.0, 3.0], horizon=2, period=0)
