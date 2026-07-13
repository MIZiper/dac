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
        _bulk_key: tuple = None,
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
        self._bulk_key = _bulk_key
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

    # ---- crop / downsample ----

    def decimation_step(self, target_fs: float | None) -> int:
        """Integer decimation step to approximate *target_fs* for this segment.

        Each segment uses its own sample rate, so segments of differing
        ``fs`` yield different steps.  Returns 1 when no downsampling is
        needed (``target_fs`` unset or not below this segment's ``fs``).
        """
        if target_fs is None or target_fs <= 0 or target_fs >= self.fs:
            return 1
        return max(1, int(round(self.fs / target_fs)))

    def index_bounds(self, t_start=None, t_end=None) -> tuple[int, int]:
        """Inclusive sample-index bounds covering ``[t_start, t_end]``.

        Bounds are derived arithmetically from *t0*/*dt* (no full-length
        time axis is built).  ``t_start``/``t_end`` must match the type of
        *t0*; ``None`` means unbounded on that side.  Returns ``(i0, i1)``
        with ``i0 > i1`` when the segment has no samples in range.
        """
        n = self.length
        if n == 0:
            return 0, -1
        i0, i1 = 0, n - 1
        if isinstance(self.t0, np.datetime64):
            step_ns = int(round(self.dt * 1e9)) or 1
            t0_ns = self.t0.astype("datetime64[ns]")
            if t_start is not None:
                d = (np.datetime64(t_start).astype("datetime64[ns]") - t0_ns).astype(np.int64)
                i0 = max(i0, int(-(-d // step_ns)))  # ceil division
            if t_end is not None:
                d = (np.datetime64(t_end).astype("datetime64[ns]") - t0_ns).astype(np.int64)
                i1 = min(i1, int(d // step_ns))  # floor division
        else:
            if t_start is not None:
                i0 = max(i0, int(np.ceil((t_start - self.t0) / self.dt - 1e-9)))
            if t_end is not None:
                i1 = min(i1, int(np.floor((t_end - self.t0) / self.dt + 1e-9)))
        return i0, i1

    def crop_downsample(
        self, t_start=None, t_end=None, step: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(t, y)`` for samples in ``[t_start, t_end]`` decimated by *step*.

        The time axis is computed only for the selected indices, so no
        full-resolution ``datetime64`` array is materialised for cropping.
        """
        i0, i1 = self.index_bounds(t_start, t_end)
        if i0 > i1:
            return np.array([], dtype=_time_dtype(self.t0)), np.array([])
        step = max(1, int(step))
        idx = np.arange(i0, i1 + 1, step)
        y = self.y[i0 : i1 + 1 : step]
        if isinstance(self.t0, np.datetime64):
            step_ns = int(round(self.dt * 1e9))
            t = self.t0.astype("datetime64[ns]") + idx * np.timedelta64(step_ns, "ns")
        else:
            t = self.t0 + idx * self.dt
        return t, y



class TSSegment(DataBase):
    """Time-stamped segment with explicit timestamps and direct data storage.

    Unlike :class:`TimeSegment` which computes time axes from *t0* and
    *dt*, this stores the full timestamp array directly.  Data is held in
    memory without caching or lazy loading, suitable for small timestamped
    datasets.

    The ``_t`` and ``_y`` arrays are underscore-prefixed internally so
    that :meth:`~dac.core.DataNode.get_construct_config` skips them
    during serialization.

    Parameters
    ----------
    t : np.ndarray
        Timestamp array (``datetime64`` or ``float``).
    y : np.ndarray
        Data array (same length as *t*).
    """

    def __init__(
        self,
        name: str = None,
        uuid: str = None,
        t: np.ndarray = None,
        y: np.ndarray = None,
        y_unit: str = "-",
        comment: str = "",
    ) -> None:
        super().__init__(name, uuid)
        self._t = np.asarray(t) if t is not None else np.array([])
        self._y = np.asarray(y) if y is not None else np.array([])
        self.y_unit = y_unit
        self.comment = comment

    # ---- stored arrays ----

    @property
    def t(self) -> np.ndarray:
        return self._t

    @t.setter
    def t(self, value):
        self._t = np.asarray(value)

    @property
    def y(self) -> np.ndarray:
        return self._y

    @y.setter
    def y(self, value):
        self._y = np.asarray(value)

    # ---- size ----

    @property
    def t0(self):
        """First timestamp (or ``None`` if empty)."""
        if len(self._t) == 0:
            return None
        return self._t[0]

    @property
    def t_end(self):
        """Last timestamp (or ``None`` if empty)."""
        if len(self._t) == 0:
            return None
        return self._t[-1]

    @property
    def length(self) -> int:
        return len(self._t)

    @property
    def nbytes(self) -> int:
        return self._t.nbytes + self._y.nbytes


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
        if not self._segments:
            return []
        t0_ref = self._segments[0].t0
        t_start = _coerce_time(t_start, t0_ref)
        t_end = _coerce_time(t_end, t0_ref)
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

        Each segment is cropped and downsampled independently (per its own
        sample rate) *before* merging, so segments with different ``fs`` are
        handled correctly and gap markers between segments are preserved.

        Parameters
        ----------
        t_start :
            Start of time range.  Must match the type of ``segments[0].t0``.
        t_end :
            End of time range.  Must match the type of ``segments[0].t0``.
        target_fs : float, optional
            If set, each segment is downsampled toward this sample rate (Hz).

        Returns
        -------
        t : np.ndarray
            Merged time axis (datetime64 or float64).
        y : np.ndarray
            Merged data, with ``np.nan`` markers inserted across gaps.
        dt : float
            Effective sample interval of the first segment after optional
            downsampling.  Note segments may have differing rates.
        """
        if t_start is None:
            t_start = self.time_range[0]
        if t_end is None:
            t_end = self.time_range[1]
        ref_t0 = self._segments[0].t0 if self._segments else 0.0
        t_start = _coerce_time(t_start, ref_t0)
        t_end = _coerce_time(t_end, ref_t0)

        segs = self.segments_at(t_start, t_end)
        if not segs:
            empty_t = np.array([], dtype=_time_dtype(t_start))
            return empty_t, np.array([]), 1.0

        t_parts = []
        y_parts = []
        dt = None
        prev_seg = None

        for seg in segs:
            step = seg.decimation_step(target_fs)
            t_cropped, y_cropped = seg.crop_downsample(t_start, t_end, step)

            if len(t_cropped) == 0:
                continue

            if dt is None:
                dt = seg.dt * step

            if prev_seg is not None and seg.t0 > prev_seg.t_end:
                t_parts.append(np.array([prev_seg.t_end], dtype=t_cropped.dtype))
                y_parts.append(np.array([np.nan]))

            t_parts.append(t_cropped)
            y_parts.append(y_cropped)
            prev_seg = seg

        t = np.concatenate(t_parts) if t_parts else np.array([], dtype=_time_dtype(t_start))
        y = np.concatenate(y_parts) if y_parts else np.array([])

        return t, y, dt if dt is not None else segs[0].dt


class TSChannel(DataBase):
    """Container for timestamped segments (:class:`TSSegment`) of the same channel.

    Like :class:`TimeChannel`, this groups segments sorted by *t0* and
    provides :meth:`get_merged_data` for plotting via SpecPlot.  Unlike
    ``TimeChannel``, timestamps are stored explicitly (not computed from
    *t0*+*dt*), cropping uses boolean indexing on the stored arrays, and
    downsampling uses simple integer decimation.

    ``SelectTimeRangeAction`` works visually but the downstream
    ``SetupAnalysisContextTask`` does **not** apply — ``TSSegment`` has
    no ``_cache_key`` referencing loadable files.
    """

    def __init__(
        self, name: str = None, uuid: str = None, y_unit: str = "-"
    ) -> None:
        super().__init__(name, uuid)
        self._segments: list[TSSegment] = []
        self.y_unit = y_unit

    @property
    def segments(self) -> list[TSSegment]:
        return list(self._segments)

    def add_segment(self, seg: TSSegment):
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

    def segments_at(self, t_start, t_end) -> list[TSSegment]:
        if not self._segments:
            return []
        t0_ref = self._segments[0].t0
        t_start = _coerce_time(t_start, t0_ref)
        t_end = _coerce_time(t_end, t0_ref)
        return [
            s
            for s in self._segments
            if s.t0 is not None and s.t_end is not None
            and s.t0 < t_end and s.t_end > t_start
        ]

    def unload(self):
        """Free stored data arrays (set to empty)."""
        for s in self._segments:
            s._t = np.array([])
            s._y = np.array([])

    def get_merged_data(
        self,
        t_start=None,
        t_end=None,
        target_fs: float = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Merge overlapping segments into continuous time and data arrays.

        Unlike :meth:`TimeChannel.get_merged_data` which uses arithmetic
        ``index_bounds`` and strided slicing, this method uses boolean
        masking on the stored timestamps for cropping.  Downsampling uses
        simple integer decimation computed from the mean sample interval.

        Parameters
        ----------
        t_start :
            Start of time range.  Must match the type of ``segments[0].t0``.
        t_end :
            End of time range.  Must match the type of ``segments[0].t0``.
        target_fs : float, optional
            If set, data is decimated toward this sample rate (Hz).

        Returns
        -------
        t : np.ndarray
            Merged time axis (datetime64 or float64).
        y : np.ndarray
            Merged data, with ``np.nan`` markers inserted across gaps.
        dt : float
            Approximate mean sample interval of the merged data, or 1.0
            when unavailable.
        """
        if t_start is None:
            t_start = self.time_range[0]
        if t_end is None:
            t_end = self.time_range[1]
        ref_t0 = self._segments[0].t0 if self._segments else 0.0
        t_start = _coerce_time(t_start, ref_t0)
        t_end = _coerce_time(t_end, ref_t0)

        segs = self.segments_at(t_start, t_end)
        if not segs:
            empty_t = np.array([], dtype=_time_dtype(t_start))
            return empty_t, np.array([]), 1.0

        t_parts = []
        y_parts = []
        dt = None
        prev_seg = None

        for seg in segs:
            t_seg = seg.t
            y_seg = seg.y
            if len(t_seg) == 0:
                continue

            mask = (t_seg >= t_start) & (t_seg <= t_end)
            t_cropped = t_seg[mask]
            y_cropped = y_seg[mask]

            if len(t_cropped) == 0:
                continue

            # ---- downsampling via simple decimation ----
            step = 1
            if target_fs is not None and target_fs > 0 and len(t_cropped) > 1:
                if isinstance(t_cropped[0], np.datetime64):
                    diffs = np.diff(
                        t_cropped.astype("datetime64[ns]").astype(np.int64)
                    )
                else:
                    diffs = np.diff(t_cropped.astype(np.float64))
                mean_dt_ns = np.mean(diffs)
                if mean_dt_ns > 0:
                    mean_fs = (
                        1e9 / mean_dt_ns
                        if isinstance(t_cropped[0], np.datetime64)
                        else 1.0 / mean_dt_ns
                    )
                    step = max(1, int(round(mean_fs / target_fs)))

            if step > 1:
                t_cropped = t_cropped[::step]
                y_cropped = y_cropped[::step]

            # ---- mean dt from first valid segment ----
            if dt is None and len(t_cropped) > 1:
                if isinstance(t_cropped[0], np.datetime64):
                    dt = (
                        float(
                            np.mean(
                                np.diff(
                                    t_cropped.astype("datetime64[ns]").astype(np.int64)
                                )
                            )
                        )
                        / 1e9
                    )
                else:
                    dt = float(np.mean(np.diff(t_cropped.astype(np.float64))))

            # ---- gap marker between non-contiguous segments ----
            if prev_seg is not None and seg.t0 > prev_seg.t_end:
                t_parts.append(np.array([prev_seg.t_end], dtype=t_cropped.dtype))
                y_parts.append(np.array([np.nan]))

            t_parts.append(t_cropped)
            y_parts.append(y_cropped)
            prev_seg = seg

        t = (
            np.concatenate(t_parts)
            if t_parts
            else np.array([], dtype=_time_dtype(t_start))
        )
        y = np.concatenate(y_parts) if y_parts else np.array([])

        return t, y, dt if dt is not None else 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _time_dtype(t0) -> np.dtype:
    if isinstance(t0, np.datetime64):
        return np.dtype("datetime64[ns]")
    return np.float64


def _coerce_time(val, ref):
    """Coerce *val* to match the type of *ref* for comparison.

    ``float`` epoch seconds ↔ ``np.datetime64[ns]`` are mutually
    convertible.  Returns *val* unchanged when types already match or
    when *val* is ``None``.
    """
    if val is None:
        return val
    if isinstance(ref, np.datetime64) and not isinstance(val, np.datetime64):
        return np.datetime64(int(round(float(val) * 1e9)), "ns")
    if not isinstance(ref, np.datetime64) and isinstance(val, np.datetime64):
        return val.astype("datetime64[ns]").astype(np.int64) / 1e9
    return val


def normalize_time(val):
    """Coerce *val* to ``np.datetime64`` or ``float`` for comparison.

    Accepts ``None``, ``str`` (ISO datetime or numeric), ``float``,
    and ``np.datetime64``. Empty string → ``None``.
    """
    if val is None or val == "":
        return None
    if isinstance(val, np.datetime64):
        return val
    if isinstance(val, str):
        try:
            return np.datetime64(val)
        except ValueError:
            return float(val)
    return float(val)


def time_to_str(val):
    """Convert a time value to a clean string for YAML persistence.

    ``np.datetime64`` → ISO string, ``float`` → numeric string,
    ``None`` → ``""``.
    """
    if val is None:
        return ""
    if isinstance(val, np.datetime64):
        ts = val.astype("datetime64[ms]")
        return str(ts).replace("T", " ")
    return str(val)
