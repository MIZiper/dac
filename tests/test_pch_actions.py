import numpy as np
import pytest

from dac.core import GCK
from dac.modules.pch import TimeChannel, TimeSegment
from dac.modules.pch.plots import compute_windowed_xy_map


def _ch(name, t0, dt, y, unit="-"):
    ch = TimeChannel(name=name, y_unit=unit)
    seg = TimeSegment(name=name, t0=t0, length=len(y), dt=dt, y_unit=unit)
    seg._y = np.asarray(y, dtype=float)
    ch.add_segment(seg)
    return ch


class TestWindowedXYMap:
    def test_duration_map(self):
        x = _ch("x", 0.0, 1.0, np.arange(10))
        y = _ch("y", 0.0, 1.0, np.arange(10))
        xe, ye, v, c = compute_windowed_xy_map(
            x, y, avg_seconds=2.0, x_bins=[0, 10, 5], y_bins=[0, 10, 5]
        )
        assert v.shape == (2, 2)
        assert np.isclose(np.nansum(v), 10.0)
        assert np.isclose(v[0, 0], 6.0)
        assert np.isclose(v[1, 1], 4.0)
        assert np.isnan(v[0, 1]) and np.isnan(v[1, 0])
        assert c[0, 0] == 3 and c[1, 1] == 2

    def test_z_statistics(self):
        x = _ch("x", 0.0, 1.0, np.zeros(4))
        y = _ch("y", 0.0, 1.0, np.zeros(4))
        z = _ch("z", 0.0, 1.0, np.arange(4), unit="N")
        kw = dict(
            z_ch=z,
            avg_seconds=2.0,
            x_bins=[-0.5, 0.5, 1.0],
            y_bins=[-0.5, 0.5, 1.0],
        )
        # per-window z means are 0.5 and 2.5
        _, _, mean_v, _ = compute_windowed_xy_map(x, y, statistic="mean", **kw)
        _, _, min_v, _ = compute_windowed_xy_map(x, y, statistic="min", **kw)
        _, _, max_v, _ = compute_windowed_xy_map(x, y, statistic="max", **kw)
        _, _, std_v, _ = compute_windowed_xy_map(x, y, statistic="std", **kw)
        _, _, rms_v, _ = compute_windowed_xy_map(x, y, statistic="rms", **kw)
        assert np.isclose(mean_v[0, 0], 1.5)
        assert np.isclose(min_v[0, 0], 0.5)
        assert np.isclose(max_v[0, 0], 2.5)
        assert np.isclose(std_v[0, 0], 1.0)
        assert np.isclose(rms_v[0, 0], np.sqrt((0.25 + 6.25) / 2))

    def test_overlap_clipped(self):
        base = np.datetime64("2024-01-01T00:00:00")
        x = _ch("x", base + np.timedelta64(10, "s"), 0.5, np.arange(20))
        y = _ch("y", base, 1.0, np.arange(30))
        _, _, v, _ = compute_windowed_xy_map(
            x, y, avg_seconds=2.0, x_bins=[0, 20, 2], y_bins=[0, 30, 3]
        )
        # overlap is 10 s (x from +10s to +20s)
        assert np.isclose(np.nansum(v), 10.0)

    def test_datetime_float_equivalence(self):
        base = np.datetime64("2024-01-01T00:00:00")
        xd = _ch("x", base, 1.0, np.arange(10))
        yd = _ch("y", base, 1.0, np.arange(10))
        xf = _ch("x", 0.0, 1.0, np.arange(10))
        yf = _ch("y", 0.0, 1.0, np.arange(10))
        kw = dict(avg_seconds=2.0, x_bins=[0, 10, 5], y_bins=[0, 10, 5])
        *_, vd, _ = compute_windowed_xy_map(xd, yd, **kw)
        *_, vf, _ = compute_windowed_xy_map(xf, yf, **kw)
        np.testing.assert_allclose(vd, vf)

    def test_multiple_segments_with_gap(self):
        x = TimeChannel(name="x")
        for t0, vals in [(0.0, np.zeros(5)), (10.0, np.zeros(5))]:
            seg = TimeSegment(name="x", t0=t0, length=5, dt=1.0)
            seg._y = vals
            x.add_segment(seg)
        y = _ch("y", 0.0, 1.0, np.zeros(15))
        _, _, v, _ = compute_windowed_xy_map(
            x, y, avg_seconds=1.0, x_bins=[-0.5, 0.5, 1.0], y_bins=[-0.5, 0.5, 1.0]
        )
        # only the 10 covered seconds count; the 5-second gap is skipped
        assert np.isclose(v[0, 0], 10.0)

    def test_auto_edges(self):
        x = _ch("x", 0.0, 1.0, np.linspace(0, 100, 101))
        y = _ch("y", 0.0, 1.0, np.linspace(0, 50, 101))
        xe, ye, v, _ = compute_windowed_xy_map(x, y, avg_seconds=5.0)
        assert len(xe) == 51 and len(ye) == 51
        assert xe[0] < xe[-1] and ye[0] < ye[-1]

    def test_progress_steps(self):
        x = _ch("x", 0.0, 1.0, np.arange(10))
        y = _ch("y", 0.0, 1.0, np.arange(10))
        steps = []
        compute_windowed_xy_map(
            x, y, avg_seconds=2.0, on_step=lambda i, n: steps.append((i, n))
        )
        assert [s for s, _ in steps] == [1, 2, 3]
        assert all(n == 3 for _, n in steps)

    def test_invalid_statistic(self):
        x = _ch("x", 0.0, 1.0, np.arange(4))
        with pytest.raises(ValueError):
            compute_windowed_xy_map(x, x, statistic="bogus")

    def test_invalid_bins(self):
        x = _ch("x", 0.0, 1.0, np.arange(4))
        with pytest.raises(ValueError):
            compute_windowed_xy_map(x, x, x_bins=[0, 1], y_bins=[0, 1])


class TestXYStatisticPlotAction:
    def test_smoke_with_z(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from dac.modules.pch.actions import XYStatisticPlotAction

        x = _ch("x", 0.0, 1.0, np.arange(10), unit="rpm")
        y = _ch("y", 0.0, 1.0, np.arange(10), unit="Nm")
        z = _ch("z", 0.0, 1.0, np.arange(10) * 2, unit="deg")

        act = XYStatisticPlotAction(GCK)
        fig = Figure()
        FigureCanvasAgg(fig)
        act._figure = fig

        act(
            x,
            y,
            z_channel=z,
            avg_seconds=2.0,
            x_bins=[0, 10, 5],
            y_bins=[0, 10, 5],
            statistic="mean",
        )
        # main axes + colorbar axes
        assert len(fig.axes) >= 2
        assert "rpm" in fig.axes[0].get_xlabel()
        assert "Nm" in fig.axes[0].get_ylabel()
