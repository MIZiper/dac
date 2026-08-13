"""Actions for the event_log module.

Provides interactive range-selection for creating event log entries,
visualisation with SpecPlot + event overlays, and statistical extraction.
"""

import numpy as np

from dac.core.actions import ActionBase, VAB, PAB, SAB, TAB
from dac.modules.pch import TimeChannel, TSChannel
from dac.modules.pch.actions import SelectTimeRangeAction, SpecPlotAction

from . import (
    EventLogCollection,
    EventStatistics,
    parse_time,
)
from .plots import overlay_events


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _compute_stats(y_clean: np.ndarray, wanted: list[str]) -> dict:
    """Compute *wanted* statistics on a clean (non-NaN) array."""
    record: dict = {}
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
    return record


def _nearest_sample(ch: TimeChannel, t):
    """Return ``(t_sample, value)`` of the sample nearest to time *t*.

    Uses the nearest non-NaN sample; returns ``(None, None)`` when the
    channel has no data.
    """
    t_axis, y, _ = ch.get_merged_data()
    if len(t_axis) == 0:
        return None, None

    idx = int(np.searchsorted(t_axis, t))
    best = None
    for j in (idx - 1, idx, idx + 1):
        if 0 <= j < len(t_axis):
            val = y[j]
            if np.isnan(val):
                continue
            dist = abs(t_axis[j] - t)
            if best is None or dist < best[0]:
                best = (dist, t_axis[j], val)

    if best is None:
        return None, None
    return best[1], best[2]


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
        event_data: list = None,
    ) -> EventLogCollection:
        coll = EventLogCollection(name="EventLog")
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


class SelectEventRangeAction(SelectTimeRangeAction):
    """Interactive time-range selection for adding event log entries.

    Reuses the interaction of
    :class:`~dac.modules.pch.actions.SelectTimeRangeAction`; only the
    hint/title text and the :attr:`setup_handler` differ.
    """

    CAPTION = "Select range to add event log"

    _SUPTITLE = "Drag to select range  |  Right-click → Add event log"
    _HINT = (
        "Selection recorded.  Right-click this action "
        "in the Action panel → 'Add event log entry'"
    )

    setup_handler: "Callable[['SelectEventRangeAction'], None] | None" = None


# ---------------------------------------------------------------------------
# InspectTimeRangeAction — quick view of a range / point (quick_action)
# ---------------------------------------------------------------------------


class InspectTimeRangeAction(SelectTimeRangeAction):
    """Interactive selection for quick-inspecting channel data.

    Reuses the interaction of
    :class:`~dac.modules.pch.actions.SelectTimeRangeAction`.  On
    right-click the installed :attr:`setup_handler` (usually
    ``InspectSelectionTask``) reads the selection and shows a table:
    statistics per channel for a range, or the nearest sample value per
    channel for a single point.  *stats* selects which statistics to
    compute for the range case.
    """

    CAPTION = "Inspect time range"

    _SUPTITLE = "Drag to select range  |  Click for a point  |  Right-click → Inspect"
    _HINT = (
        "Selection recorded.  Right-click this action "
        "in the Action panel → 'Inspect selection'"
    )

    setup_handler: "Callable[['InspectTimeRangeAction'], None] | None" = None

    def __call__(
        self,
        channels: list[TimeChannel | TSChannel],
        target_fs: float = 1.0,
        stats: str = "mean,std,min,max,rms",
    ):
        self._stats = stats
        super().__call__(channels, target_fs=target_fs)


class OverlayEventsAction(VAB):
    """Draw event range spans and labels on a figure rendered by SpecPlot.

    Used as the second sub-action inside
    :class:`PlotEventsAction` (SAB).
    """

    CAPTION = "Overlay event ranges"

    def __call__(
        self,
        events: list[EventLogCollection],
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

        result = EventStatistics(name="Event Stats")
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
                    "event": entry.name,
                    "t_start": entry.start,
                    "t_end": entry.end,
                    "n_samples": int(len(y_clean)),
                }
                record.update(_compute_stats(y_clean, wanted))

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

        if len(stat_names) == 1:
            stat_name = stat_names[0]
            channels = []
            events = []
            for rec in statistics.records:
                if rec["channel"] not in channels:
                    channels.append(rec["channel"])
                if rec["event"] not in events:
                    events.append(rec["event"])

            value_map = {
                (rec["channel"], rec["event"]): rec.get(stat_name, "")
                for rec in statistics.records
            }
            data = [
                [value_map.get((ch, ev), "") for ch in channels]
                for ev in events
            ]
            self.present({
                "title": f"Event Statistics - {stat_name}",
                "headers": {"row": events, "col": channels},
                "data": data,
            })
        else:
            row_labels = []
            data = []
            for rec in statistics.records:
                row_labels.append(f"{rec['channel']}/{rec['event']}")
                data.append([rec.get(s, "") for s in stat_names])

            self.present({
                "title": "Event Statistics",
                "headers": {"row": row_labels, "col": stat_names},
                "data": data,
            })
