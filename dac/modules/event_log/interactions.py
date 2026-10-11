"""Live interactions for the event_log module.

These plug into :class:`~dac.core.interact.InteractivePlotAction` so an
event-log workflow (overlay + add + inspect) can be switched on top of an
already rendered plot without re-plotting.

No PyQt dependency is imported at module level, so this module stays
importable in headless / web environments; dialogs are imported lazily
inside the dialog-driven tool only.
"""

from dac.core.interact import OverlayBase, RangeSelectTool
from dac.modules.pch import TimeChannel, TSChannel, time_to_str

from . import EventLogCollection
from .plots import overlay_events


def _time_channels(ctx) -> list:
    """All time-like channels visible from the context (pch channels)."""
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
        return super().available(ctx) and bool(ctx.find_nodes(EventLogCollection))

    def _resolve_events(self) -> list:
        """Collections to overlay.

        ``events: null`` means "all EventLogCollection currently in
        context"; an explicit list (including ``[]``) is used as-is.
        """
        if self.events is not None:
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
    """Select a range and append it to an existing/new event log group.

    Requires a host that can show dialogs (PyQt); unavailable on the web
    front-end.
    """

    CAPTION = "Add event log"
    _HINT = "Drag to select a range  |  Right-click → Add event log entry"
    REQUIRES_DIALOG = True

    @classmethod
    def available(cls, ctx) -> bool:
        return super().available(ctx) and bool(_time_channels(ctx))

    def on_confirm(self) -> None:
        if self.t_start is None:
            return
        ui = self.ctx.ui
        container = self.ctx.container
        if ui is None or container is None:
            return

        from .tasks import _AddEventLogDialog, _group_name_of
        from .actions import CreateEventLogAction

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
            target = CreateEventLogAction(context_key=container.current_key)
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

