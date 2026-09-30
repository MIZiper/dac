"""Live interactions for the event_log module.

These plug into :class:`~dac.core.interact.InteractivePlotAction` so an
event-log workflow (overlay + add + inspect) can be switched on top of an
already rendered plot without re-plotting.

No PyQt dependency is imported at module level, so this module stays
importable in headless environments; the dialogs are imported lazily.
"""

import numpy as np

from dac.core.interact import OverlayBase, RangeSelectTool
from dac.modules.pch import TimeChannel, TSChannel, time_to_str

from . import EventLogCollection
from .actions import _compute_stats, _nearest_sample
from .plots import overlay_events


def _time_channels(ctx) -> list:
    """All time-like channels visible from the context (TimeChannel + TSChannel)."""
    return list(ctx.find_nodes(TimeChannel)) + list(ctx.find_nodes(TSChannel))


class EventRangesOverlay(OverlayBase):
    """Draw event ranges of one or more ``EventLogCollection`` as spans."""

    CAPTION = "Event ranges"
    DEFAULT_ON = True

    def __init__(
        self,
        ctx,
        events: list[EventLogCollection] = None,
        label_axes_index: int = 0,
    ) -> None:
        super().__init__(ctx)
        self.events = events
        self.label_axes_index = label_axes_index

    @classmethod
    def available(cls, ctx) -> bool:
        return bool(ctx.find_nodes(EventLogCollection))

    def _resolve_events(self) -> list:
        if self.events:
            return list(self.events)
        return list(self.ctx.find_nodes(EventLogCollection))

    def attach(self) -> None:
        artists = overlay_events(
            self._resolve_events(),
            self.label_axes_index,
            figure=self.ctx.figure,
        )
        for artist in artists:
            self.track(artist)


class AddEventLogTool(RangeSelectTool):
    """Select a range and append it to an existing/new event log group."""

    CAPTION = "Add event log"
    _HINT = "Drag to select a range  |  Right-click → Add event log entry"

    @classmethod
    def available(cls, ctx) -> bool:
        return bool(_time_channels(ctx))

    def on_confirm(self) -> None:
        if self.t_start is None:
            return
        from .tasks import _AddEventLogDialog, _group_name_of
        from .actions import CreateEventLogAction

        container = self.ctx.container
        ui = self.ctx.ui
        if container is None or ui is None:
            return

        t_start, t_end = self.t_start, self.t_end
        dlg = _AddEventLogDialog(t_start, t_end, container, parent=ui)
        if not dlg.exec_():
            return

        group_name, label = dlg.result()
        time_str = f"{time_to_str(t_start)} ~ {time_to_str(t_end)}"

        target = None
        for act in container.actions:
            if isinstance(act, CreateEventLogAction) and _group_name_of(act) == group_name:
                target = act
                break

        if target is None:
            target = CreateEventLogAction(context_key=self.ctx.container.current_key)
            target.container = container
            target.get_construct_config()
            target.out_name = group_name
            target._construct_config["event_data"] = []
            target.status = CreateEventLogAction.ActionStatus.CONFIGURED
            container.actions.append(target)
            ui.message(f"Created event log group '{group_name}'")

        if not isinstance(target._construct_config.get("event_data"), list):
            target._construct_config["event_data"] = []
        target._construct_config["event_data"].append([time_str, label])
        target.status = CreateEventLogAction.ActionStatus.CONFIGURED

        if hasattr(ui, "action_list_widget"):
            ui.action_list_widget.refresh()
        ui.message(f"Added event '{label}' to group '{group_name}'")


class InspectTool(RangeSelectTool):
    """Select a range/point and show a statistics table of the channels."""

    CAPTION = "Inspect selection"
    _HINT = "Drag to select a range  |  Click for a point  |  Right-click → Inspect"

    def __init__(self, ctx, stats: str = "mean,std,min,max,rms") -> None:
        super().__init__(ctx)
        self.stats = stats

    @classmethod
    def available(cls, ctx) -> bool:
        return bool(_time_channels(ctx))

    def on_confirm(self) -> None:
        ui = self.ctx.ui
        if ui is None or self.t_start is None:
            return
        channels = _time_channels(self.ctx)
        if not channels:
            return

        if self.t_start == self.t_end:
            table = self._point_table(channels, self.t_start)
        else:
            table = self._range_table(channels, self.t_start, self.t_end, self.stats)
        if hasattr(ui, "show_stats"):
            ui.show_stats(table)

    def _range_table(self, channels, t_start, t_end, stats):
        from .actions import ExtractEventStatisticsAction

        wanted = [s.strip() for s in stats.split(",") if s.strip()]
        wanted = [
            s for s in ExtractEventStatisticsAction._AVAILABLE_STATS if s in wanted
        ]

        rows, data = [], []
        for ch in channels:
            _t, y, _dt = ch.get_merged_data(t_start=t_start, t_end=t_end)
            y_clean = y[~np.isnan(y)]
            if len(y_clean) == 0:
                continue
            rec = _compute_stats(y_clean, wanted)
            rows.append(f"{ch.name} [{ch.y_unit}]")
            data.append([rec.get(s, "") for s in wanted])

        return {
            "title": f"Statistics  {time_to_str(t_start)} → {time_to_str(t_end)}",
            "headers": {"row": rows, "col": wanted},
            "data": data,
        }

    def _point_table(self, channels, t):
        rows, data = [], []
        for ch in channels:
            t_sample, value = _nearest_sample(ch, t)
            rows.append(f"{ch.name} [{ch.y_unit}]")
            if t_sample is None:
                data.append(["", ""])
            else:
                data.append([time_to_str(t_sample), value])

        return {
            "title": f"Values at  {time_to_str(t)}",
            "headers": {"row": rows, "col": ["time", "value"]},
            "data": data,
        }
