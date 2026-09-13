import numpy as np
import pytest

from dac.core import GCK
from dac.modules.pch import TimeChannel, TimeSegment
from dac.modules.event_log import EventLogCollection
from dac.modules.event_log.actions import (
    ExtractStableEventsAction,
    InspectTimeRangeAction,
    _compute_stats,
    _nearest_sample,
)
from dac.modules.event_log.stability import (
    intersect_intervals,
    rolling_std,
    stable_intervals,
)


def _channel(y, t0=0.0, dt=0.1, unit="m/s2", name="Ch"):
    seg = TimeSegment(name="seg", t0=t0, length=len(y), dt=dt, y_unit=unit)
    seg._y = np.asarray(y, dtype=float)
    ch = TimeChannel(name=name, y_unit=unit)
    ch.add_segment(seg)
    return ch


class TestComputeStats:
    def test_values(self):
        rec = _compute_stats(
            np.array([1.0, 2.0, 3.0, 4.0]),
            ["mean", "std", "min", "max", "rms"],
        )
        assert rec["mean"] == pytest.approx(2.5)
        assert rec["std"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0]))
        assert rec["min"] == 1.0
        assert rec["max"] == 4.0
        assert rec["rms"] == pytest.approx(np.sqrt(np.mean(np.arange(1.0, 5.0) ** 2)))

    def test_subset_only(self):
        rec = _compute_stats(np.array([1.0, 2.0]), ["mean", "rms"])
        assert set(rec.keys()) == {"mean", "rms"}


class TestNearestSample:
    def test_float_nearest(self):
        ch = _channel(np.arange(10, dtype=float))
        t, v = _nearest_sample(ch, 0.26)
        assert t == pytest.approx(0.3)
        assert v == 3.0

    def test_float_before_first(self):
        ch = _channel(np.arange(10, dtype=float))
        t, v = _nearest_sample(ch, -0.5)
        assert t == pytest.approx(0.0)
        assert v == 0.0

    def test_datetime_axis(self):
        t0 = np.datetime64("2024-01-01T00:00:00")
        ch = _channel(np.array([10.0, 20.0, 30.0]), t0=t0, dt=1.0)
        target = t0 + np.timedelta64(2, "s")
        t, v = _nearest_sample(ch, target)
        assert t == target
        assert v == 30.0

    def test_empty_channel(self):
        ch = TimeChannel(name="empty")
        t, v = _nearest_sample(ch, 0.0)
        assert t is None and v is None


class TestInspectActionSignature:
    def test_stats_param(self):
        sig = InspectTimeRangeAction._SIGNATURE
        assert "channels" in sig.parameters
        assert "stats" in sig.parameters
        assert sig.parameters["stats"].default == "mean,std,min,max,rms"
        assert InspectTimeRangeAction.setup_handler is None


class TestRollingStd:
    def test_matches_numpy(self):
        from numpy.testing import assert_allclose

        y = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
        std, ok = rolling_std(y, 3)
        assert not ok[0] and not ok[1]
        assert ok[2] and std[2] == pytest.approx(0.0)
        assert ok[5] and std[5] == pytest.approx(0.0)
        assert std[3] == pytest.approx(np.std([1.0, 1.0, 5.0]))
        assert_allclose(std[2:], [np.std(y[i - 2 : i + 1]) for i in range(2, 6)])

    def test_nan_window_unstable(self):
        y = np.array([1.0, np.nan, 1.0])
        std, ok = rolling_std(y, 1)
        assert ok[0] and not ok[1] and ok[2]
        std3, ok3 = rolling_std(y, 3)
        assert not np.any(ok3)


class TestStableIntervals:
    def test_flat_then_noisy(self):
        rng = np.random.default_rng(0)
        y = np.concatenate([np.zeros(100), rng.normal(0, 10, 100)])
        ch = _channel(y, t0=0.0, dt=0.01)
        ivs = stable_intervals(ch, tolerance=0.01, win_seconds=0.1, min_duration=0.1)
        assert len(ivs) == 1
        s, e = ivs[0]
        assert s == pytest.approx(0.09)
        assert e == pytest.approx(0.99)

    def test_min_duration_filters(self):
        ch = _channel(np.zeros(20), t0=0.0, dt=0.1)
        assert stable_intervals(ch, 0.1, 0.1, min_duration=3.0) == []
        assert len(stable_intervals(ch, 0.1, 0.1, min_duration=1.0)) == 1

    def test_datetime_axis(self):
        t0 = np.datetime64("2024-01-01T00:00:00")
        ch = _channel(np.zeros(50), t0=t0, dt=0.1)
        ivs = stable_intervals(ch, 0.1, 0.1, min_duration=0.0)
        assert len(ivs) == 1
        # epoch seconds for 2024-01-01
        assert ivs[0][0] == pytest.approx(float(t0.astype("datetime64[ns]").astype(np.int64)) / 1e9)


class TestIntersectIntervals:
    def test_overlap(self):
        a = [(0, 5), (10, 20)]
        b = [(3, 12), (15, 25)]
        assert intersect_intervals([a, b]) == [(3, 5), (10, 12), (15, 20)]

    def test_disjoint(self):
        assert intersect_intervals([[(0, 1)], [(2, 3)]]) == []

    def test_nested(self):
        assert intersect_intervals([[(0, 10)], [(2, 4)]]) == [(2, 4)]

    def test_empty_input_is_empty(self):
        assert intersect_intervals([[(0, 5)], []]) == []


class TestExtractStableEventsAction:
    def test_common_interval(self):
        a = _channel(np.zeros(50), t0=0.0, dt=0.1, name="A")
        b = _channel(np.zeros(50), t0=0.0, dt=0.1, name="B")
        act = ExtractStableEventsAction(GCK)
        coll = act(
            [a, b],
            tolerances={"A": 0.01, "B": 0.01},
            win_seconds=0.2,
            min_duration=0.5,
        )
        assert isinstance(coll, EventLogCollection)
        assert len(coll.entries) == 1
        assert coll.entries[0].name == "Stable_1"

    def test_intersection_shrinks_to_shortest(self):
        a = _channel(np.zeros(50), dt=0.1, name="A")
        b_y = np.concatenate([np.zeros(30), np.arange(20, dtype=float)])
        b = _channel(b_y, dt=0.1, name="B")
        act = ExtractStableEventsAction(GCK)
        coll = act(
            [a, b],
            tolerances={"A": 0.01, "B": 0.01},
            win_seconds=0.2,
            min_duration=0.0,
        )
        assert len(coll.entries) == 1
        assert float(coll.entries[0].end) == pytest.approx(3.0)

    def test_per_channel_tolerance(self):
        a = _channel(np.zeros(50), dt=0.1, name="A")
        b = _channel(np.tile([10.0, -10.0], 25), dt=0.1, name="B")
        act = ExtractStableEventsAction(GCK)
        # B alternates by +/-10, so std is 10: unstable at tol 0.01, stable at 100
        strict = act([a, b], tolerances={"A": 0.01, "B": 0.01}, win_seconds=0.2, min_duration=0.0)
        loose = act([a, b], tolerances={"A": 0.01, "B": 100.0}, win_seconds=0.2, min_duration=0.0)
        assert len(strict.entries) == 0
        assert len(loose.entries) == 1

    def test_fallback_tolerance(self):
        a = _channel(np.zeros(20), dt=0.1, name="A")
        act = ExtractStableEventsAction(GCK)
        coll = act([a], tolerance=0.1, win_seconds=0.1, min_duration=0.0)
        assert len(coll.entries) == 1

    def test_missing_tolerance(self):
        a = _channel(np.zeros(20), dt=0.1, name="A")
        act = ExtractStableEventsAction(GCK)
        coll = act([a], tolerances={}, tolerance=None)
        assert len(coll.entries) == 0

    def test_datetime_output(self):
        from dac.modules.event_log import parse_time

        t0 = np.datetime64("2024-01-01T00:00:00")
        a = _channel(np.zeros(30), t0=t0, dt=0.1, name="A")
        act = ExtractStableEventsAction(GCK)
        coll = act([a], tolerance=0.1, win_seconds=0.1, min_duration=0.0)
        assert len(coll.entries) == 1
        assert parse_time(coll.entries[0].start) == t0

