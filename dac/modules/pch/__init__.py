"""Defines `TimeSegment` and `TimeChannel` for acquisition data management.

TimeSegment is like TimeData but with absolute time positioning (t0) and
lazy loading of the y array via a Loader. TimeChannel groups segments of
the same measurement channel together.
"""

import bisect

from dac.core.data import DataBase
import numpy as np


class TimeSegment(DataBase):
    """Time-series segment with absolute time positioning and lazy loading.

    The `y` array is loaded on first access via the associated Loader and
    process-wide Cache. Call `unload()` to release memory when no longer needed.
    """

    def __init__(
        self,
        name: str = None,
        uuid: str = None,
        t0: float = 0.0,
        duration: float = 0.0,
        dt: float = 1.0,
        y_unit: str = "-",
        comment: str = "",
        _cache_key: tuple = None,
        _loader=None,
    ) -> None:
        super().__init__(name, uuid)
        self.t0 = t0
        self.duration = duration
        self.dt = dt
        self.y_unit = y_unit
        self.comment = comment
        self._y: np.ndarray | None = None
        self._cache_key = _cache_key
        self._loader = _loader

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

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    @property
    def length(self) -> int:
        if self._y is not None:
            return len(self._y)
        return int(self.duration / self.dt)

    @property
    def nbytes(self) -> int:
        if self._y is not None:
            return self._y.nbytes
        return 0

    @property
    def t(self) -> np.ndarray:
        return self.t0 + np.arange(self.length) * self.dt

    def unload(self):
        self._y = None


class TimeChannel(DataBase):
    """Container for TimeSegments of the same measurement channel.

    Segments are sorted by their start time. Provides methods to query
    segments by time range and merge data for plotting or extraction.
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
    def time_range(self) -> tuple[float, float]:
        segs = self.segments
        if not segs:
            return (0.0, 0.0)
        t0 = min(s.t0 for s in segs)
        t1 = max(s.t0 + s.duration for s in segs)
        return (t0, t1)

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self._segments)

    @property
    def is_fully_loaded(self) -> bool:
        return all(s.is_loaded for s in self._segments)

    def segments_at(self, t_start: float, t_end: float) -> list[TimeSegment]:
        return [
            s
            for s in self.segments
            if s.t0 < t_end and s.t0 + s.duration > t_start
        ]

    def unload(self):
        for s in self._segments:
            s.unload()

    def get_merged_data(
        self,
        t_start: float = None,
        t_end: float = None,
        target_fs: float = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Merge overlapping segments into continuous time and data arrays.

        Parameters
        ----------
        t_start : float, optional
            Start of time range (absolute epoch seconds).
        t_end : float, optional
            End of time range (absolute epoch seconds).
        target_fs : float, optional
            If set, downsample output to this sample rate (Hz).

        Returns
        -------
        t : np.ndarray
            Merged absolute time axis.
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
            return np.array([]), np.array([]), 1.0

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

            if prev_seg is not None:
                prev_end = prev_seg.t0 + prev_seg.duration
                if seg.t0 > prev_end:
                    t_parts.append(np.array([prev_end]))
                    y_parts.append(np.array([np.nan]))

            t_parts.append(t_cropped)
            y_parts.append(y_cropped)
            prev_seg = seg

        t = np.concatenate(t_parts) if t_parts else np.array([])
        y = np.concatenate(y_parts) if y_parts else np.array([])

        if target_fs is not None and target_fs < segs[0].fs:
            interval = int(round(segs[0].fs / target_fs))
            if interval > 1:
                t = t[::interval]
                y = y[::interval]
                dt = dt * interval

        return t, y, dt
