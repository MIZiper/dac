import numpy as np
import pytest

from dac.modules.pch import TimeChannel, TimeSegment
from dac.modules.event_log.actions import (
    InspectTimeRangeAction,
    _compute_stats,
    _nearest_sample,
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
