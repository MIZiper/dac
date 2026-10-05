"""Live interactions for the PCH module.

Switchable range-selection tools for an already rendered channel plot:
statistics over a dragged range (or a single point) and creating a new
analysis context from a selection.

PyQt is imported lazily (only when a dialog-driven tool actually runs),
so this module stays headless-safe and importable by the web front-end.
"""

import numpy as np

from dac.core.interact import RangeSelectTool
from dac.modules.pch import TimeChannel, TSChannel, TimeSegment, time_to_str
from dac.modules.timedata import TimeData


_STATS = ("mean", "std", "min", "max", "rms")
_DEFAULT_STATS = ",".join(_STATS)


def _time_channels(ctx) -> list:
    """All time-like channels visible from the context."""
    return (
        list(ctx.find_nodes(TimeChannel))
        + list(ctx.find_nodes(TSChannel))
        + list(ctx.find_nodes(TimeData))
    )


def _parse_stats(stats: str) -> list[str]:
    wanted = [s.strip() for s in (stats or "").split(",") if s.strip()]
    return [s for s in _STATS if s in wanted]


def _range_stats(channel, t_start, t_end, wanted) -> dict:
    """Compute *wanted* statistics for *channel* over ``[t_start, t_end]``.

    Works for pch TimeChannel/TSChannel (``get_merged_data``) and
    timedata TimeData (``x`` / ``y`` arrays).
    """
    if hasattr(channel, "get_merged_data"):
        _t, y, _dt = channel.get_merged_data(t_start=t_start, t_end=t_end)
    else:
        x = np.asarray(channel.x)
        mask = (x >= t_start) & (x <= t_end)
        y = np.asarray(channel.y)[mask]

    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    record: dict = {}
    for name in wanted:
        try:
            if name == "mean":
                record[name] = float(np.mean(y))
            elif name == "std":
                record[name] = float(np.std(y))
            elif name == "min":
                record[name] = float(np.min(y))
            elif name == "max":
                record[name] = float(np.max(y))
            elif name == "rms":
                record[name] = float(np.sqrt(np.mean(y ** 2)))
        except (ValueError, FloatingPointError):
            record[name] = float("nan")
    return record


def _nearest_in_segment(seg, moment):
    """Nearest non-NaN sample of one segment to *moment* (or ``(None,None)``)."""
    if isinstance(seg, TimeSegment):
        y = seg.y
        n = len(y)
        if n == 0:
            return None, None
        t0, dt = seg.t0, seg.dt
        if isinstance(t0, np.datetime64):
            d = (
                np.datetime64(moment).astype("datetime64[ns]")
                - t0.astype("datetime64[ns]")
            ).astype(np.int64)
            idx = int(round(d / (dt * 1e9)))
            t_val = t0 + np.timedelta64(int(round(idx * dt * 1e9)), "ns")
        else:
            idx = int(round((moment - t0) / dt))
            t_val = t0 + idx * dt
        idx = min(max(idx, 0), n - 1)
        if np.isnan(y[idx]):
            return None, None
        return t_val, y[idx]

    # TSSegment: explicit timestamp array
    t, y = seg.t, seg.y
    if len(t) == 0:
        return None, None
    idx = int(np.searchsorted(t, moment))
    best = None
    for j in (idx - 1, idx, idx + 1):
        if 0 <= j < len(t) and not np.isnan(y[j]):
            dist = abs(t[j] - moment)
            if best is None or dist < best[0]:
                best = (dist, t[j], y[j])
    if best is None:
        return None, None
    return best[1], best[2]


def nearest_sample(channel, moment):
    """Nearest non-NaN ``(time, value)`` of *channel* to *moment*.

    Looks only inside the segment(s) that contain *moment*; when *moment*
    falls in a gap between segments ``(None, None)`` is returned.
    """
    if moment is None:
        return None, None

    if not hasattr(channel, "segments"):
        x = np.asarray(channel.x)
        y = np.asarray(channel.y)
        if len(x) == 0:
            return None, None
        idx = int(np.searchsorted(x, moment))
        best = None
        for j in (idx - 1, idx, idx + 1):
            if 0 <= j < len(x) and not np.isnan(y[j]):
                dist = abs(x[j] - moment)
                if best is None or dist < best[0]:
                    best = (dist, x[j], y[j])
        if best is None:
            return None, None
        return best[1], best[2]

    for seg in channel.segments:
        t0, t_end = seg.t0, seg.t_end
        if t0 is None or t_end is None:
            continue
        if t0 <= moment <= t_end:
            return _nearest_in_segment(seg, moment)
    return None, None


class RangeStatsTool(RangeSelectTool):
    """Drag (range) or click (point) to show statistics for the channels.

    A dragged range shows the selected statistics per channel; a single
    click shows the nearest sample value per channel.  A click landing in a
    gap between segments leaves that channel blank.
    """

    CAPTION = "Range statistics"
    _HINT = "Drag to select a range  |  Click for a point"

    def __init__(
        self,
        ctx,
        channels: list[TimeChannel | TSChannel | TimeData] = None,
        stats: str = _DEFAULT_STATS,
        plot_dt: float = None,
    ) -> None:
        super().__init__(ctx)
        self.channels = channels
        self.stats = stats
        self.plot_dt = plot_dt

    @classmethod
    def available(cls, ctx) -> bool:
        return super().available(ctx) and bool(_time_channels(ctx))

    def param_options(self, name: str) -> list:
        if name == "channels":
            return _time_channels(self.ctx)
        return []

    # -- selection handling ------------------------------------------------

    def _announce_selection(self) -> None:
        if self.t_start is None or self.t_end is None:
            return
        ui = self.ctx.ui
        if ui is None or not hasattr(ui, "show_stats"):
            return
        channels = (
            self.channels if self.channels is not None else _time_channels(self.ctx)
        )
        if not channels:
            return

        if self.t_start == self.t_end:
            table = self._point_table(channels, self.t_start)
        else:
            table = self._range_table(channels, self.t_start, self.t_end, self.stats)
        ui.show_stats(table)

    def _range_table(self, channels, t_start, t_end, stats):
        wanted = _parse_stats(stats)
        rows, data = [], []
        for ch in channels:
            rec = _range_stats(ch, t_start, t_end, wanted)
            rows.append(f"{ch.name} [{getattr(ch, 'y_unit', '-')}]")
            data.append([f"{rec.get(s, float('nan')):.4g}" for s in wanted])

        return {
            "title": f"Statistics  {time_to_str(t_start)} → {time_to_str(t_end)}",
            "headers": {"row": rows, "col": wanted},
            "data": data,
        }

    def _point_table(self, channels, t):
        rows, data = [], []
        for ch in channels:
            t_sample, value = nearest_sample(ch, t)
            rows.append(f"{ch.name} [{getattr(ch, 'y_unit', '-')}]")
            if t_sample is None:
                data.append(["", ""])
            else:
                data.append([time_to_str(t_sample), value])

        return {
            "title": f"Values at  {time_to_str(t)}",
            "headers": {"row": rows, "col": ["time", "value"]},
            "data": data,
        }


class SelectContextTool(RangeSelectTool):
    """Drag over a channel plot to create a new analysis context.

    Requires a host that can show dialogs (PyQt); unavailable on the web
    front-end, which only supports non-dialog interactions.
    """

    CAPTION = "Select → context"
    _HINT = "Drag to select a range  |  Right-click → Setup Analysis Context"
    REQUIRES_DIALOG = True

    @classmethod
    def available(cls, ctx) -> bool:
        return super().available(ctx) and bool(_time_channels(ctx))

    def on_confirm(self) -> None:
        if self.t_start is None:
            return
        ui = self.ctx.ui
        if ui is None:
            return
        from dac.modules.pch.tasks import SetupAnalysisContextTask

        task = SetupAnalysisContextTask(dac_win=ui, name="Setup Analysis Context")
        task.current_context = self.ctx.current_context
        self._t_start = self.t_start
        self._t_end = self.t_end
        self._channels = _time_channels(self.ctx)
        task(self)
