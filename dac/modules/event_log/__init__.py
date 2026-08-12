"""Event log module for annotating time-series data with labeled event ranges.

Defines `EventLogEntry`, `EventLogCollection`, and `EventStatistics` as
DataBase subclasses that integrate with the DAC data context and action
resolution system.
"""

import numpy as np

from dac.core.data import DataBase


class EventLogEntry(DataBase):
    """A single event defined by a time range and optional colour.

    The entry's ``name`` (inherited from `DataNode`) serves as its label.
    All attributes are ``str`` (basic types) so they auto-serialise
    through the standard `DataNode` mechanism.
    """

    def __init__(
        self,
        name: str = None,
        uuid: str = None,
        start: str = "",
        end: str = "",
        color: str = "",
    ) -> None:
        super().__init__(name, uuid)
        self.start = start
        self.end = end
        self.color = color


class EventLogCollection(DataBase):
    """Named collection of `EventLogEntry` nodes stored as children.

    ``color`` can be set to override the auto-assigned colour used
    when rendering overlays.
    """

    def __init__(self, name: str = None, uuid: str = None) -> None:
        super().__init__(name, uuid)
        self.color = ""

    @property
    def entries(self) -> list[EventLogEntry]:
        return [c for c in self.children if isinstance(c, EventLogEntry)]

    def add_entry(
        self, start: str, end: str, label: str = "", color: str = ""
    ) -> EventLogEntry:
        entry = EventLogEntry(
            name=label or f"Event_{len(self.entries)}",
            start=str(start),
            end=str(end),
            color=color,
        )
        self.add_child(entry)
        return entry

    def remove_entry(self, entry: EventLogEntry) -> None:
        self.remove_child(entry.name)


class EventStatistics(DataBase):
    """Per-event statistical summary (transient — not persisted to file).

    ``_records`` stores a list of dicts, each holding channel name, event
    label, time range, and the requested statistical measures.
    """

    def __init__(self, name: str = None, uuid: str = None) -> None:
        super().__init__(name, uuid)
        self._records: list[dict] = []

    @property
    def records(self) -> list[dict]:
        return self._records


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def parse_time(s: str):
    """Parse a time string to ``float`` or ``np.datetime64``.

    Returns ``None`` for empty / unparseable input.
    """
    if not s or not s.strip():
        return None
    s = s.strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    try:
        return np.datetime64(s)
    except (ValueError, TypeError):
        pass
    return None


def time_midpoint(a, b):
    """Midpoint between two time values (float or datetime64)."""
    try:
        return (float(a) + float(b)) / 2.0
    except (ValueError, TypeError):
        return a + (b - a) / 2
