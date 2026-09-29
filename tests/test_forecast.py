"""Forecasting tests.

Coherence is an exact arithmetic property, and leakage is a structural one, so both
can be *asserted* rather than measured. That is unusual in machine learning and worth
exploiting when it is available.
"""

from __future__ import annotations

import math

import pytest

from forecast.backtest import (
    BacktestError,
    compare,
    mase,
    naive_scale,
    rolling_origin,
    smape,
)
from forecast.hierarchy import (
    Hierarchy,
    HierarchyError,
    bottom_up,
    historical_proportions,
    optimal,
    top_down,
)
from forecast.models import croston, drift, moving_average, naive, naive_seasonal


def tree() -> Hierarchy:
    h = Hierarchy()
    h.add("total")
    h.add("north", parent="total")
    h.add("south", parent="total")
    h.add("n1", parent="north")
    h.add("n2", parent="north")
    h.add("s1", parent="south")
    return h


# Deliberately incoherent: 400 + 550 != 1000, and 180 + 190 != 400.
BASE = {"total": 1000, "north": 400, "south": 550, "n1": 180, "n2": 190, "s1": 540}


class TestHierarchy:
    def test_structure(self):
        h = tree()
        assert h.root == "total"
        assert sorted(h.leaves()) == ["n1", "n2", "s1"]
        assert sorted(h.descendant_leaves("north")) == ["n1", "n2"]

    def test_levels_are_ordered_root_first(self):
        assert tree().levels() == [["total"], ["north", "south"], ["n1", "n2", "s1"]]

    def test_duplicate_node_is_refused(self):
        h = tree()
        with pytest.raises(HierarchyError):
            h.add("north", parent="total")

    def test_second_root_is_refused(self):
        h = tree()
        with pytest.raises(HierarchyError):
            h.add("other")

    def test_unknown_parent_is_refused(self):
        h = tree()
        with pytest.raises(HierarchyError):
            h.add("x", parent="nowhere")

    def test_independent_forecasts_are_incoherent(self):
        """The problem the whole module exists to solve."""
        assert not tree().is_coherent(BASE)


class TestReconciliation:
    @pytest.mark.parametrize("method", ["bottom_up", "optimal"])
    def test_every_method_produces_coherence(self, method):
        """Coherence is exact arithmetic, so it is asserted, not measured."""
        h = tree()
        result = {"bottom_up": bottom_up, "optimal": optimal}[method](h, BASE)
        assert h.is_coherent(result)

    def test_top_down_is_coherent(self):
        h = tree()
        result = top_down(h, BASE, {"n1": 0.2, "n2": 0.2, "s1": 0.6})
        assert h.is_coherent(result)

    def test_bottom_up_preserves_the_leaves_exactly(self):
        h = tree()
        result = bottom_up(h, BASE)
        assert result["n1"] == BASE["n1"]
        assert result["total"] == BASE["n1"] + BASE["n2"] + BASE["s1"]

    def test_top_down_preserves_the_root_exactly(self):
        h = tree()
        result = top_down(h, BASE, {"n1": 0.2, "n2": 0.2, "s1": 0.6})
        assert result["total"] == pytest.approx(BASE["total"])

    def test_optimal_lands_between_the_two_extremes(self):
        """It uses both levels, so it should agree with neither exactly."""
        h = tree()
        result = optimal(h, BASE)
        assert bottom_up(h, BASE)["total"] < result["total"] < BASE["total"]

    def test_zero_weight_everywhere_degenerates_to_bottom_up(self):
        """Ignoring every parent's own forecast is exactly bottom-up, so this is a
        check that the weighting means what it claims."""
        h = tree()
        zero = optimal(h, BASE, weights=dict.fromkeys(h.nodes, 0.0))
        assert zero == pytest.approx(bottom_up(h, BASE))

    def test_a_larger_child_absorbs_more_of_the_adjustment(self):
        h = tree()
        base = {"total": 200, "north": 0, "south": 0, "n1": 90, "n2": 10, "s1": 0}
        result = optimal(h, base)
        assert result["n1"] > result["n2"]

    def test_proportions_from_history(self):
        h = tree()
        props = historical_proportions(h, {"n1": [1, 1], "n2": [1, 1], "s1": [2, 2]})
        assert props["s1"] == pytest.approx(0.5)
        assert sum(props.values()) == pytest.approx(1.0)

    def test_no_history_falls_back_to_an_equal_split(self):
        """Better than dividing by zero or silently returning nothing."""
        h = tree()
        props = historical_proportions(h, {"n1": [], "n2": [], "s1": []})
        assert all(v == pytest.approx(1 / 3) for v in props.values())

    def test_top_down_rejects_useless_proportions(self):
        with pytest.raises(HierarchyError):
            top_down(tree(), BASE, {"n1": 0.0, "n2": 0.0, "s1": 0.0})

    def test_a_deep_tree_stays_coherent(self):
        h = Hierarchy()
        h.add("root")
        for i in range(3):
            h.add(f"l1-{i}", parent="root")
            for j in range(3):
                h.add(f"l2-{i}{j}", parent=f"l1-{i}")
        base = {name: 10.0 + idx for idx, name in enumerate(h.nodes)}
        assert h.is_coherent(optimal(h, base))


class TestModels:
    def test_naive_repeats_the_last_value(self):
        assert naive([1, 2, 3], 3) == [3.0, 3.0, 3.0]

    def test_seasonal_naive_repeats_the_season(self):
        assert naive_seasonal([1, 2, 3, 4], 4, period=4) == [1.0, 2.0, 3.0, 4.0]

    def test_seasonal_naive_falls_back_when_history_is_short(self):
        assert naive_seasonal([5], 2, period=4) == [5.0, 5.0]

    def test_moving_average(self):
        assert moving_average([2, 4, 6, 8], 1, window=2) == [7.0]

    def test_drift_follows_a_trend(self):
        assert drift([10, 20, 30], 1)[0] == pytest.approx(40.0)

    def test_empty_history_is_handled(self):
        assert naive([], 3) == [0.0, 0.0, 0.0]
        assert croston([], 3) == [0.0, 0.0, 0.0]

    def test_croston_handles_intermittent_demand(self):
        """Averaging a series that is 80% zeros forecasts 0.3 units for something
        only ever ordered in whole boxes."""
        series = [0, 0, 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10]
        result = croston(series, 1)[0]
        # Demand of 10 every 5 periods is a rate near 2 per period.
        assert 1.0 < result < 4.0

    def test_croston_beats_a_moving_average_on_the_rate(self):
        series = [0, 0, 0, 0, 20] * 6
        # A moving average over the tail is dominated by whichever zeros it lands on.
        assert croston(series, 1)[0] > moving_average(series[:-1], 1, window=4)[0]

    def test_croston_with_a_single_observation(self):
        assert croston([0, 0, 7], 2) == [7.0, 7.0]

    def test_all_zero_demand(self):
        assert croston([0, 0, 0, 0], 2) == [0.0, 0.0]


class TestMetrics:
    def test_smape_is_bounded(self):
        assert smape([1.0], [1_000_000.0]) <= 2.0

    def test_smape_treats_zero_against_zero_as_no_error(self):
        """The case that makes MAPE divide by zero."""
        assert smape([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_mase_below_one_means_better_than_doing_nothing(self):
        train = [10, 20, 10, 20, 10, 20]
        assert mase([10.0], [10.0], train) < 1.0

    def test_mase_above_one_means_worse_than_doing_nothing(self):
        train = [10, 20, 10, 20, 10, 20]
        assert mase([10.0], [500.0], train) > 1.0

    def test_mase_scale_denominator_is_in_sample(self):
        """Using the test period would let the difficulty of the test set flatter
        the model."""
        assert naive_scale([10, 20, 10, 20], period=1) == pytest.approx(10.0)

    def test_a_flat_series_reports_not_comparable_rather_than_perfect(self):
        """Reporting 0.0 would claim perfection where there is nothing to compare to."""
        assert math.isnan(mase([5.0], [5.0], [5, 5, 5, 5]))


class TestBacktest:
    def test_the_forecaster_never_sees_the_future(self):
        """The test that matters. A backtest with leakage reports an accuracy the
        system will never achieve in production, and reports it confidently."""
        series = list(range(40))
        seen: list[list[float]] = []

        def spy(history, horizon):
            seen.append(list(history))
            return [history[-1]] * horizon

        rolling_origin(series, spy, horizon=4, initial=10)

        for window in seen:
            # Every value handed over must precede the values it is asked to predict.
            assert window == series[: len(window)]
            assert max(window) < series[len(window)]

    def test_the_window_expands(self):
        series = list(range(40))
        lengths = []

        def spy(history, horizon):
            lengths.append(len(history))
            return [0.0] * horizon

        rolling_origin(series, spy, horizon=4, initial=10, step=5)
        assert lengths == sorted(lengths) and lengths[0] == 10

    def test_a_forecaster_returning_the_wrong_length_is_an_error(self):
        with pytest.raises(ValueError):
            rolling_origin(list(range(40)), lambda h, n: [0.0], horizon=4)

    def test_a_mutating_forecaster_cannot_corrupt_later_folds(self):
        series = list(range(40))

        def vandal(history, horizon):
            history.clear()
            return [0.0] * horizon

        assert rolling_origin(series, vandal, horizon=4, initial=10).summary()["folds"] > 1

    def test_a_series_too_short_to_backtest_raises_instead_of_a_silent_empty_result(self):
        # A caller that reads .summary()["folds"] without checking it is 0 would
        # otherwise report every metric on this as if it meant something.
        with pytest.raises(BacktestError, match="one fold needs"):
            rolling_origin([1, 2, 3], naive, horizon=4)

    def test_seasonal_naive_beats_naive_on_seasonal_data(self):
        """The comparison the project exists to make — before reaching for anything
        more complicated.

        The series carries a little drift on top of the season. A *perfectly* periodic
        series has zero seasonal-naive error, which makes MASE undefined — correctly,
        since there is nothing to compare against — so a fixture that clean would test
        the wrong thing.
        """
        pattern = [10, 20, 30, 40]
        series = [pattern[i % 4] + i * 0.1 for i in range(40)]
        scores = compare(
            series,
            {
                "naive": naive,
                "seasonal": lambda h, n: naive_seasonal(h, n, period=4),
            },
            horizon=4,
            initial=12,
            period=4,
        )
        assert scores["seasonal"]["mae"] < scores["naive"]["mae"]

        # MASE's denominator *is* seasonal naive, so seasonal naive scores 1.0 by
        # definition — it cannot beat itself. That is the point of the metric: any
        # model worth deploying has to come in under this line.
        assert scores["seasonal"]["mase"] == pytest.approx(1.0, abs=0.01)
        assert scores["naive"]["mase"] > 1.0
        # Each model's own default benchmark (seasonal naive, since period=4 was
        # passed) beats plain naive on data that actually has a season.
        assert scores["naive"]["beats_benchmark"] is False

    def test_all_forecasters_are_scored_on_identical_folds(self):
        """Otherwise the comparison measures the split rather than the models."""
        series = [10, 20, 30, 40] * 10
        scores = compare(
            series,
            {"a": naive, "b": lambda h, n: moving_average(h, n)},
            horizon=4,
            initial=12,
        )
        assert scores["a"]["folds"] == scores["b"]["folds"]
