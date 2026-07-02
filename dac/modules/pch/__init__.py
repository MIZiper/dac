"""Defines `TimeSegment` and `TimeChannel` for acquisition data management.

TimeSegment is like TimeData but with absolute time positioning (t0) and
lazy loading of the y array via a Loader. TimeChannel groups segments of
the same measurement channel together.

t0 supports both ``np.datetime64`` (absolute timestamps) and ``float``
(relative seconds from an arbitrary origin). All time-axis properties
preserve the original type.
"""

import bisect

from dac.core.data import DataBase
import numpy as np


class TimeSegment(DataBase):
    """Time-series segment with absolute time positioning and lazy loading.

    The `y` array is loaded on first access via the associated Loader and
    process-wide Cache. Call `unload()` to release memory when no longer needed.

    Parameters
    ----------
    t0 : np.datetime64 or float
        Start time. ``np.datetime64`` gives absolute time; ``float`` gives
        relative seconds.
    length : int
        Number of samples in the segment.
    """

    def __init__(
        self,
        name: str = None,
        uuid: str = None,
        t0: "np.datetime64 | float" = 0.0,
        length: int = 0,
        dt: float = 1.0,
        y_unit: str = "-",
        comment: str = "",
        _cache_key: tuple = None,
        _loader=None,
    ) -> None:
        super().__init__(name, uuid)
        self.t0 = t0
        self._length = length
        self.dt = dt
        self.y_unit = y_unit
        self.comment = comment
        self._y: np.ndarray | None = None
        self._cache_key = _cache_key
        self._loader = _loader

    # ---- loaded data ----

    @property
    def y(self) -> np.ndarray:
        if self._y is None and self._loader is not None and self._cache_key is not None:
            self._y = self._loader.load_full(self._cache_key, self)
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def is_loaded(self) -> bool:
        return self._y is not None

    def unload(self):
        self._y = None

    # ---- size ----

    @property
    def length(self) -> int:
        if self._y is not None:
            return len(self._y)
        return self._length

    @length.setter
    def length(self, value: int):
        self._length = value

    @property
    def duration(self) -> float:
        """Duration in seconds (``length * dt``)."""
        return self.length * self.dt

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    @property
    def nbytes(self) -> int:
        if self._y is not None:
            return self._y.nbytes
        return 0

    # ---- time axis ----

    @property
    def t(self) -> np.ndarray:
        n = self.length
        if n == 0:
            return np.array([], dtype=_time_dtype(self.t0))
        if isinstance(self.t0, np.datetime64):
            step_ns = int(round(self.dt * 1e9))
            return self.t0 + np.arange(n) * np.timedelta64(step_ns, "ns")
        return self.t0 + np.arange(n, dtype=np.float64) * self.dt

    @property
    def t_end(self):
        """End time boundary (``t0 + duration``), matching the type of *t0*."""
        dur = self.duration
        if isinstance(self.t0, np.datetime64):
            return self.t0 + np.timedelta64(int(round(dur * 1e9)), "ns")
        return self.t0 + dur


class TimeChannel(DataBase):
    """Container for TimeSegments of the same measurement channel.

    Segments are kept sorted by *t0*. Provides methods to query segments
    by time range and merge data for plotting or extraction.
    """

    def __init__(
        self, name: str = None, uuid: str = None, y_unit: str = "-"
    ) -> None:
        super().__init__(name, uuid)
        self._segments: list[TimeSegment] = []
        self.y_unit = y_unit

    @property
    def segments(self) -> list[TimeSegment]:
        return list(self._segments)

    def add_segment(self, seg: TimeSegment):
        bisect.insort(self._segments, seg, key=lambda s: s.t0)
        self.add_child(seg)
        if self.y_unit == "-" and seg.y_unit != "-":
            self.y_unit = seg.y_unit

    @property
    def time_range(self) -> tuple:
        segs = self._segments
        if not segs:
            t0 = 0.0
            return (t0, t0)
        return (segs[0].t0, segs[-1].t_end)

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self._segments)

    @property
    def is_fully_loaded(self) -> bool:
        return all(s.is_loaded for s in self._segments)

    def segments_at(self, t_start, t_end) -> list[TimeSegment]:
        return [
            s
            for s in self._segments
            if s.t0 < t_end and s.t_end > t_start
        ]

    def unload(self):
        for s in self._segments:
            s.unload()

    def get_merged_data(
        self,
        t_start=None,
        t_end=None,
        target_fs: float = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Merge overlapping segments into continuous time and data arrays.

        Parameters
        ----------
        t_start :
            Start of time range.  Must match the type of ``segments[0].t0``.
        t_end :
            End of time range.  Must match the type of ``segments[0].t0``.
        target_fs : float, optional
            If set, downsample output to this sample rate (Hz).

        Returns
        -------
        t : np.ndarray
            Merged time axis (datetime64 or float64).
        y : np.ndarray
            Merged data.
        dt : float
            Effective sample interval after optional downsampling.
        """
        if t_start is None:
            t_start = self.time_range[0]
        if t_end is None:
            t_end = self.time_range[1]

        segs = self.segments_at(t_start, t_end)
        if not segs:
            empty_t = np.array([], dtype=_time_dtype(t_start))
            return empty_t, np.array([]), 1.0

        t_parts = []
        y_parts = []
        dt = segs[0].dt
        prev_seg = None

        for seg in segs:
            t_seg = seg.t
            y_seg = seg.y
            mask = (t_seg >= t_start) & (t_seg <= t_end)
            t_cropped = t_seg[mask]
            y_cropped = y_seg[mask]

            if len(t_cropped) == 0:
                continue

            if prev_seg is not None and seg.t0 > prev_seg.t_end:
                t_parts.append(np.array([prev_seg.t_end]))
                y_parts.append(np.array([np.nan]))

            t_parts.append(t_cropped)
            y_parts.append(y_cropped)
            prev_seg = seg

        t = np.concatenate(t_parts) if t_parts else np.array([], dtype=_time_dtype(t_start))
        y = np.concatenate(y_parts) if y_parts else np.array([])

        if target_fs is not None and target_fs < segs[0].fs:
            interval = int(round(segs[0].fs / target_fs))
            if interval > 1:
                t = t[::interval]
                y = y[::interval]
                dt = dt * interval

        return t, y, dt


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _time_dtype(t0) -> np.dtype:
    if isinstance(t0, np.datetime64):
        return np.dtype("datetime64[ns]")
    return np.float64
