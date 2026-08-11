"""Actions for the event_log module.

Provides interactive range-selection for creating event log entries,
visualisation with SpecPlot + event overlays, and statistical extraction.
"""

import warnings

import numpy as np
from matplotlib import gridspec

from dac.core.actions import ActionBase, VAB, PAB, SAB, TAB
from dac.modules.pch import TimeChannel, TSChannel, time_to_str
from dac.modules.pch.actions import SpecPlotAction
from dac.modules.pch.plots import is_datetime_type, setup_datetime_axis
from dac.modules.pch.spec import spec_from_dict, render_spec

from . import (
    EventLogEntry,
    EventLogCollection,
    EventStatistics,
    parse_time,
)
from .plots import overlay_events


# ---------------------------------------------------------------------------
# CreateEventLogAction — pure data producer
# ---------------------------------------------------------------------------


class CreateEventLogAction(ActionBase):
    """Create an EventLogCollection from a YAML-friendly event_data list.

    ``event_data`` format::

        - - "2024-01-01 10:00:00 ~ 2024-01-01 10:00:05"
          - label 1
        - - "12.3 ~ 45.6"
          - label 2

    Each entry is ``[time_range_str, label]``.  The time range uses ``~``
    as the delimiter.  The action is independent of channel data.
    """

    CAPTION = "Create event log collection"

    def __call__(
        self,
        group_name: str = "",
        event_data: list = None,
    ) -> EventLogCollection:
        name = self.out_name or group_name or "EventLog"
        coll = EventLogCollection(name=name)
        if event_data:
            for item in event_data:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                time_range_str = str(item[0])
                label = str(item[1])
                parts = time_range_str.split("~", 1)
                if len(parts) == 2:
                    s0, s1 = parts[0].strip(), parts[1].strip()
                else:
                    s0, s1 = "", ""
                coll.add_entry(s0, s1, label)
        return coll


# ---------------------------------------------------------------------------
# SelectEventRangeAction — interactive range selection (quick_action)
# ---------------------------------------------------------------------------


class SelectEventRangeAction(VAB):
    """Interactive time-range selection on TimeChannel previews.

    **Left-drag** selects a time range; **right-click** triggers the
    :attr:`setup_handler` (typically an :class:`~dac.modules.event_log.tasks.AddEventLogTask`)
    to add the selected range as an event log entry to a
    :class:`CreateEventLogAction`.

    Interaction mirrors :class:`~dac.modules.pch.actions.SelectTimeRangeAction`.
    """

    CAPTION = "Select range to add event log"

    setup_handler: "Callable[['SelectEventRangeAction'], None] | None" = None

    def __call__(
        self,
        channels: list[TimeChannel | TSChannel],
        target_fs: float = 1.0,
    ) -> None:
        if not channels:
            return

        self._channels = channels
        self._t_start = None
        self._t_end = None
        self._press_xdata = None
        self._span_patches = []
        self._vlines = []
        self._axes = []
        self._dragging = False

        fig = self.figure
        fig.suptitle(
            "Drag to select range  |  Right-click → Add event log"
        )

        unit_groups: list[tuple[str, list[TimeChannel]]] = []
        seen_units: dict[str, int] = {}
        for ch in channels:
            idx = seen_units.get(ch.y_unit)
            if idx is None:
                idx = len(unit_groups)
                unit_groups.append((ch.y_unit, []))
                seen_units[ch.y_unit] = idx
            unit_groups[idx][1].append(ch)

        n_rows = len(unit_groups)
        gs = gridspec.GridSpec(n_rows, 1, figure=fig)
        gs.update(hspace=0.05)
        axes = []
        datetime_setup = False

        self._all_times = []
        all_y_flat = []

        for i, (unit, grp) in enumerate(unit_groups):
            ax = fig.add_subplot(gs[i], sharex=axes[0] if axes else None)
            axes.append(ax)

            for ch in grp:
                t, y, _ = ch.get_merged_data(target_fs=target_fs)
                if len(t) == 0:
                    continue

                if not datetime_setup and is_datetime_type(t):
                    setup_datetime_axis(ax)
                    datetime_setup = True

                ax.plot(t, y, label=f"{ch.name}")
                self._all_times.append(t)
                all_y_flat.extend(y if len(y) else [0])

            ax.set_ylabel(f"[{unit}]")
            ax.legend(loc="upper right", fontsize="small")
            if i < n_rows - 1:
                ax.tick_params(labelbottom=False)

        if not datetime_setup and axes:
            axes[-1].set_xlabel("Time [s]")

        self._axes = axes
        self._y_min = float(np.min(all_y_flat)) if all_y_flat else 0
        self._y_max = float(np.max(all_y_flat)) if all_y_flat else 0
        self._y_span = self._y_max - self._y_min or 1.0

        canvas = self.canvas

        _ref_type = None
        for ch in channels:
            if ch._segments:
                _ref_type = type(ch._segments[0].t0)
                break
        _is_datetime = _ref_type is np.datetime64

        def _normalize_x(x):
            if x is None:
                return None
            if _is_datetime:
                import matplotlib.dates as _mdates

                dt = _mdates.num2date(x).replace(tzinfo=None)
                return np.datetime64(dt.isoformat())
            return x

        def _ax_for(event):
            if event.inaxes is not None and event.inaxes in axes:
                return event.inaxes
            return None

        def _show_hint():
            _clear_hint()
            hint = fig.text(
                0.5, 0.01,
                "Selection recorded.  Right-click this action "
                "in the Action panel → 'Add event log entry'",
                ha="center", fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="lightyellow",
                    alpha=0.9,
                ),
            )
            self._hint = hint
            canvas.draw_idle()

        def _clear_hint():
            if hasattr(self, "_hint") and self._hint is not None:
                self._hint.remove()
                self._hint = None

        def on_press(event):
            if canvas.widgetlock.locked():
                return
            ax = _ax_for(event)
            if ax is None:
                return
            if event.button == 1:
                self._press_xdata = event.xdata
                self._t_start = _normalize_x(event.xdata)
                self._dragging = True
                _clear_spans()
                _clear_vlines()
                for ax_i in axes:
                    span = ax_i.axvspan(
                        event.xdata, event.xdata,
                        alpha=0.2, color="green",
                    )
                    self._span_patches.append(span)
                canvas.draw_idle()

        def on_motion(event):
            if not self._dragging:
                return
            x = _normalize_x(event.xdata)
            if x is None:
                return
            t0 = min(self._t_start, x)
            t1 = max(self._t_start, x)
            for span in self._span_patches:
                span.remove()
            self._span_patches.clear()
            for ax_i in axes:
                span = ax_i.axvspan(t0, t1, alpha=0.2, color="green")
                self._span_patches.append(span)
            canvas.draw_idle()

        def on_release(event):
            if not self._dragging and event.button != 3:
                return
            if event.button == 3:
                if not canvas.widgetlock.locked():
                    _on_right_click()
                return
            self._dragging = False
            x = _normalize_x(event.xdata)
            if self._t_start is None or x is None:
                _clear_spans()
                return
            _clear_spans()
            t_range = axes[0].get_xlim()
            click_threshold = (t_range[1] - t_range[0]) * 0.005
            if abs(event.xdata - self._press_xdata) < click_threshold:
                self._t_end = self._t_start
                for ax_i in axes:
                    vline = ax_i.axvline(
                        self._t_start, color="red", linestyle="--",
                    )
                    self._vlines.append(vline)
            else:
                self._t_end = x
                t0, t1 = sorted([self._t_start, self._t_end])
                self._t_start, self._t_end = t0, t1
                for ax_i in axes:
                    span = ax_i.axvspan(t0, t1, alpha=0.15, color="green")
                    self._span_patches.append(span)
            canvas.draw_idle()
            _show_hint()

        def _clear_spans():
            for s in self._span_patches:
                s.remove()
            self._span_patches.clear()

        def _clear_vlines():
            for v in self._vlines:
                v.remove()
            self._vlines.clear()

        def _on_right_click():
            if self._t_start is None:
                return
            _clear_hint()
            handler = SelectEventRangeAction.setup_handler
            if handler is None:
                warnings.warn(
                    "SelectEventRangeAction.setup_handler is not set; "
                    "cannot add event log entry from selection.",
                    stacklevel=2,
                )
                return
            handler.current_context = self.container.CurrentContext
            handler(self)

        self._cids.append(canvas.mpl_connect("button_press_event", on_press))
        self._cids.append(canvas.mpl_connect("motion_notify_event", on_motion))
        self._cids.append(canvas.mpl_connect("button_release_event", on_release))


# ---------------------------------------------------------------------------
# OverlayEventsAction — overlay event spans on existing axes
# ---------------------------------------------------------------------------


class OverlayEventsAction(VAB):
    """Draw event range spans and labels on a figure rendered by SpecPlot.

    Used as the second sub-action inside
    :class:`PlotEventsAction` (SAB).
    """

    CAPTION = "Overlay event ranges"

    def __call__(
        self,
        events: list[EventLogCollection] = None,
        label_axes_index: int = 0,
    ) -> None:
        if not events:
            return
        overlay_events(events, label_axes_index, figure=self.figure)


# ---------------------------------------------------------------------------
# PlotEventsAction — SAB: SpecPlot + Overlays
# ---------------------------------------------------------------------------


class PlotEventsAction(SAB, seq=[SpecPlotAction, OverlayEventsAction]):
    """Render TimeChannels via SpecPlot and overlay event log ranges.

    Config example (YAML in the SAB parameter editor)::

        SpecPlotAction:
          channels: [AccelX, AccelY]
          spec: {layout: [ax1, ax2], axes: {ax1: {chs: [AccelX]}, ...}}
        OverlayEventsAction:
          events: [MyLogGroup]
          label_axes_index: 0
    """

    CAPTION = "Plot events on channel"


# ---------------------------------------------------------------------------
# ExtractEventStatisticsAction — per-event statistics
# ---------------------------------------------------------------------------


class ExtractEventStatisticsAction(PAB):
    """Extract per-event statistics from TimeChannels.

    For each channel and each event range in *events*, the data within
    the range is fetched via ``get_merged_data`` and the requested
    statistics are computed.  All results are merged into a single
    :class:`EventStatistics` node.
    """

    CAPTION = "Extract event statistics"

    # sentinel so stat-names iterate in a fixed, predictable order
    _AVAILABLE_STATS = ("mean", "std", "min", "max", "rms")

    def __call__(
        self,
        channels: list[TimeChannel],
        events: EventLogCollection,
        stats: str = "mean,std,min,max,rms",
    ) -> EventStatistics:
        wanted = [s.strip() for s in stats.split(",") if s.strip()]
        wanted = [s for s in self._AVAILABLE_STATS if s in wanted]

        result = EventStatistics(name=self.out_name or "Event Stats")
        total = len(channels) * len(events.entries)
        count = 0

        for ch in channels:
            for entry in events.entries:
                t0 = parse_time(entry.start)
                t1 = parse_time(entry.end)
                if t0 is None or t1 is None:
                    continue

                t, y, _dt = ch.get_merged_data(t_start=t0, t_end=t1)
                if len(y) == 0:
                    continue

                y_clean = y[~np.isnan(y)]
                if len(y_clean) == 0:
                    continue

                record: dict = {
                    "channel": ch.name,
                    "event": entry.label or entry.name,
                    "t_start": entry.start,
                    "t_end": entry.end,
                    "n_samples": int(len(y_clean)),
                }
                for name in wanted:
                    try:
                        if name == "mean":
                            record[name] = float(np.mean(y_clean))
                        elif name == "std":
                            record[name] = float(np.std(y_clean))
                        elif name == "min":
                            record[name] = float(np.min(y_clean))
                        elif name == "max":
                            record[name] = float(np.max(y_clean))
                        elif name == "rms":
                            record[name] = float(np.sqrt(np.mean(y_clean ** 2)))
                    except (ValueError, FloatingPointError):
                        record[name] = float("nan")

                result._records.append(record)
                count += 1
                self.progress(count, total)

        self.message(
            f"Extracted statistics for {len(result._records)} event × channel combinations"
        )
        return result


# ---------------------------------------------------------------------------
# ViewEventStatisticsAction — table display
# ---------------------------------------------------------------------------


class ViewEventStatisticsAction(TAB):
    """Display event statistics as a table."""

    CAPTION = "View event statistics"

    def __call__(self, statistics: EventStatistics) -> None:
        if not statistics.records:
            self.message("No statistics data to display.")
            return

        from collections import OrderedDict

        ordered = OrderedDict()
        for s in ExtractEventStatisticsAction._AVAILABLE_STATS:
            if s in statistics.records[0]:
                ordered[s] = None
        stat_names = list(ordered.keys())

        col_labels = []
        data = []
        for rec in statistics.records:
            col_labels.append(f"{rec['channel']}/{rec['event']}")
            data.append([rec.get(s, "") for s in stat_names])

        self.present({
            "title": "Event Statistics",
            "headers": {"row": stat_names, "col": col_labels},
            "data": list(zip(*data)),
        })
