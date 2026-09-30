"""Live interactions for the PCH module.

Switchable range-selection tools for an already rendered channel plot:
statistics over a dragged range and creating a new analysis context from
a selection.  PyQt is imported lazily so the module stays headless-safe.
"""

import numpy as np

from dac.core.interact import RangeSelectTool
from dac.modules.pch import TimeChannel, TSChannel
from dac.modules.timedata import TimeData


def _time_channels(ctx) -> list:
    return (
        list(ctx.find_nodes(TimeChannel))
        + list(ctx.find_nodes(TSChannel))
        + list(ctx.find_nodes(TimeData))
    )


_STATS = ("mean", "std", "min", "max", "rms")


def _range_stats(channel, t_start, t_end) -> dict:
    """Compute basic statistics for *channel* over ``[t_start, t_end]``.

    Works for pch TimeChannel/TSChannel (``get_merged_data``) and
    timedata TimeData (``x`` / ``y`` arrays).
    """
    if hasattr(channel, "get_merged_data"):
        _t, y, _dt = channel.get_merged_data(t_start=t_start, t_end=t_end)
    else:
        x = np.asarray(channel.x)
        mask = (x >= t_start) & (x <= t_end)
        y = np.asarray(channel.y)[mask]

    y = y[~np.isnan(y)]
    record = {}
    for name in _STATS:
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


class RangeStatsTool(RangeSelectTool):
    """Drag over a channel plot to show statistics for the selected range."""

    CAPTION = "Range statistics"
    _HINT = "Drag to select a time range for statistics"

    def __init__(
        self,
        ctx,
        channels: list = None,
        plot_dt: float = None,
    ) -> None:
        super().__init__(ctx)
        self.channels = channels
        self.plot_dt = plot_dt

    @classmethod
    def available(cls, ctx) -> bool:
        return bool(_time_channels(ctx))

    def _announce_selection(self) -> None:
        if self.t_start is None or self.t_end is None:
            return
        if self.t_start == self.t_end:
            return
        ui = self.ctx.ui
        if ui is None or not hasattr(ui, "show_stats"):
            return

        channels = self.channels or _time_channels(self.ctx)
        if not channels:
            return
        _present_statistics(ui.show_stats, channels, t_range=(self.t_start, self.t_end))


class SelectContextTool(RangeSelectTool):
    """Drag over a channel plot to create a new analysis context."""

    CAPTION = "Select → context"
    _HINT = "Drag to select a range  |  Right-click → Setup Analysis Context"

    @classmethod
    def available(cls, ctx) -> bool:
        return bool(_time_channels(ctx))

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


def _present_statistics(renderer, channels: list, t_range=None) -> None:
    cols = []
    data = []
    for channel in channels:
        record = _range_stats(channel, t_range[0], t_range[1])
        cols.append(getattr(channel, "name", "?"))
        data.append([f"{record.get(key, float('nan')):.4g}" for key in _STATS])

    title = "Statistics"
    if t_range is not None:
        try:
            title += f" [{t_range[0]:.3f}, {t_range[1]:.3f}] s"
        except (TypeError, ValueError):
            title += f" [{t_range[0]}, {t_range[1]}]"
    renderer({
        "title": title,
        "headers": {"row": list(_STATS), "col": cols},
        "data": list(zip(*data)),
    })
