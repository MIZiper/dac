"""Plot utilities for PCH module.

Provides helpers for downsampling, time-type detection, and datetime
axis setup for TimeChannel visualization.
"""

import numpy as np
import matplotlib.dates as mdates


def is_datetime_type(t) -> bool:
    """Return True if the array has a datetime64 dtype."""
    return np.asarray(t).dtype.kind == "M"


def downsample_array(
    y: np.ndarray, src_fs: float, target_fs: float
) -> tuple[np.ndarray, float]:
    """Downsample a 1-D array by integer decimation.

    Parameters
    ----------
    y : np.ndarray
        Input data array.
    src_fs : float
        Source sample rate (Hz).
    target_fs : float
        Target sample rate (Hz). Must be <= src_fs.

    Returns
    -------
    y_ds : np.ndarray
        Strided (downsampled) array.
    dt_ds : float
        New sample interval.
    """
    if target_fs >= src_fs:
        return y, 1.0 / src_fs

    interval = int(round(src_fs / target_fs))
    if interval <= 1:
        return y, 1.0 / src_fs

    return y[::interval], 1.0 / target_fs


def downsample_time_data(
    t: np.ndarray, y: np.ndarray, target_fs: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Downsample paired time and data arrays.

    Parameters
    ----------
    t : np.ndarray
        Time axis (datetime64 or float).
    y : np.ndarray
        Data array.
    target_fs : float
        Target sample rate (Hz).

    Returns
    -------
    t_ds : np.ndarray
        Downsampled time axis.
    y_ds : np.ndarray
        Downsampled data.
    dt_ds : float
        New sample interval.
    """
    if len(t) < 2:
        return t, y, 1.0

    if is_datetime_type(t):
        src_fs = 1e9 / np.mean(np.diff(t.astype("datetime64[ns]").astype(np.int64)))
    else:
        src_fs = 1.0 / np.mean(np.diff(t))

    interval = int(round(src_fs / target_fs))
    if interval <= 1:
        return t, y, 1.0 / src_fs

    return t[::interval], y[::interval], 1.0 / target_fs


def setup_datetime_axis(ax):
    """Configure a matplotlib axis for datetime display.

    Uses ``AutoDateLocator`` and ``ConciseDateFormatter`` for
    automatic format selection based on the visible time span.
    """
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    # ax.figure.autofmt_xdate()


# ---------------------------------------------------------------------------
# Windowed XY statistic map
# ---------------------------------------------------------------------------

STATISTICS = ("mean", "min", "max", "std", "rms")


def _to_epoch_seconds(t) -> float:
    """Convert a scalar time (``datetime64`` or float seconds) to epoch seconds."""
    if isinstance(t, np.datetime64):
        return float(t.astype("datetime64[ns]").astype(np.int64)) / 1e9
    return float(t)


def _edges_from_spec(spec, dmin: float, dmax: float, default_bins: int = 50):
    """Build bin edges from a ``[start, end, step]`` spec or from a data range.

    ``spec`` may be ``None`` or a 3-element sequence.  When omitted, edges are
    auto-generated from ``[dmin, dmax]`` using *default_bins* equal bins.
    """
    if spec is not None:
        vals = list(spec)
        if len(vals) != 3:
            raise ValueError("bin spec must be [start, end, step]")
        start, end, step = (float(v) for v in vals)
        if end <= start:
            raise ValueError("bin spec requires end > start")
        if step <= 0:
            raise ValueError("bin spec requires step > 0")
        n = max(1, int(round((end - start) / step)))
        return start + np.arange(n + 1, dtype=np.float64) * step

    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
        return np.array([0.0, 1.0])
    return np.linspace(dmin, dmax, default_bins + 1)


def _accumulate_channel(ch, t0, t1, avg_seconds, nwin, tick):
    """Scan *ch* segment-by-segment, accumulating per-window count/sum/duration.

    Only samples inside ``[t0, t1]`` are used.  Each channel's samples are
    assigned to a window index by timestamp; no full time axis or merged
    array is ever materialised, so peak memory stays O(*nwin*).
    """
    from . import _coerce_time

    cnt = np.zeros(nwin, dtype=np.int64)
    tot = np.zeros(nwin, dtype=np.float64)
    dur = np.zeros(nwin, dtype=np.float64)

    for seg in ch.segments:
        tick()
        n = seg.length
        if n <= 0:
            continue

        # Coerce the requested bounds to this segment's t0 type.
        i0, i1 = seg.index_bounds(_coerce_time(t0, seg.t0), _coerce_time(t1, seg.t0))
        i0 = max(0, i0)
        i1 = min(n - 1, i1)
        y = seg.y
        if y is None:
            continue
        if len(y) < n:
            i1 = min(i1, len(y) - 1)
        if i0 > i1:
            continue

        yy = np.asarray(y[i0 : i1 + 1], dtype=np.float64)
        seg_t0 = _to_epoch_seconds(seg.t0)
        idx = np.arange(i0, i1 + 1, dtype=np.float64)
        t_rel = (seg_t0 + idx * seg.dt) - t0
        win = np.floor(t_rel / avg_seconds).astype(np.int64)

        keep = (win >= 0) & (win < nwin)
        if not np.any(keep):
            continue
        w = win[keep]
        v = yy[keep]
        finite = np.isfinite(v)
        w = w[finite]
        v = v[finite]
        if len(w) == 0:
            continue

        cnt += np.bincount(w, minlength=nwin)
        tot += np.bincount(w, weights=v, minlength=nwin)
        dur += np.bincount(w, minlength=nwin).astype(np.float64) * seg.dt

    return cnt, tot, dur


def _binned_stat(x, y, v, statistic, x_edges, y_edges):
    """Apply *statistic* over values *v* binned by (x, y)."""
    from scipy.stats import binned_statistic_2d

    if statistic == "rms":
        mean_sq = binned_statistic_2d(
            x, y, v ** 2, statistic="mean", bins=[x_edges, y_edges]
        ).statistic
        return np.sqrt(mean_sq)
    return binned_statistic_2d(
        x, y, v, statistic=statistic, bins=[x_edges, y_edges]
    ).statistic


def compute_windowed_xy_map(
    x_ch,
    y_ch,
    z_ch=None,
    avg_seconds: float = 1.0,
    x_bins=None,
    y_bins=None,
    statistic: str = "mean",
    t_start=None,
    t_end=None,
    on_step=None,
):
    """Build a windowed x-y statistic map from one or more TimeChannels.

    The time range where all channels overlap is split into fixed windows of
    *avg_seconds*.  Each window is reduced to a representative point
    ``(mean(x), mean(y))``; z, when given, is reduced to its per-window mean.

    With no *z_ch* the returned map holds the total dwell time of each x-y
    bin.  With *z_ch* it holds *statistic* (``mean``/``min``/``max``/``std``/
    ``rms``) of the per-window z means.

    Parameters
    ----------
    x_ch, y_ch : TimeChannel
        Channels providing the x and y coordinates.
    z_ch : TimeChannel, optional
        Channel whose statistic is shown as color.
    avg_seconds : float
        Averaging window length in seconds.
    x_bins, y_bins : sequence, optional
        ``[start, end, step]`` for each axis.  Auto-generated when ``None``.
    statistic : str
        One of :data:`STATISTICS`.
    t_start, t_end : optional
        Further clip the overlap range.
    on_step : callable, optional
        ``on_step(i, n)`` progress callback, called once per scanned segment
        plus once for the final binning step.

    Returns
    -------
    x_edges, y_edges : np.ndarray
    values : np.ndarray
        Statistic per bin, shape ``(len(x_edges) - 1, len(y_edges) - 1)``.
        Empty bins are ``NaN``.
    counts : np.ndarray
        Number of windows contributing to each bin (same shape as *values*).
    """
    from . import normalize_time

    if statistic not in STATISTICS:
        raise ValueError(
            f"Unknown statistic '{statistic}', expected one of {STATISTICS}"
        )
    if avg_seconds <= 0:
        raise ValueError("avg_seconds must be > 0")

    channels = [x_ch, y_ch] + ([z_ch] if z_ch is not None else [])
    total_steps = sum(len(ch.segments) for ch in channels) + 1
    _step = [0]

    def tick():
        _step[0] += 1
        if on_step is not None:
            on_step(_step[0], total_steps)

    # ---- overlap from segment metadata (no data loaded) ----
    starts = [_to_epoch_seconds(ch.time_range[0]) for ch in channels]
    ends = [_to_epoch_seconds(ch.time_range[1]) for ch in channels]
    t0 = max(starts)
    t1 = min(ends)
    if t_start is not None:
        t0 = max(t0, _to_epoch_seconds(normalize_time(t_start)))
    if t_end is not None:
        t1 = min(t1, _to_epoch_seconds(normalize_time(t_end)))

    if t1 > t0:
        nwin = max(1, int(np.ceil((t1 - t0) / avg_seconds)))
    else:
        nwin = 0

    if nwin == 0:
        for _ in range(total_steps):
            tick()
        empty = np.full((1, 1), np.nan)
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), empty, np.zeros((1, 1))

    cnt_x, sum_x, dur_x = _accumulate_channel(
        x_ch, t0, t1, avg_seconds, nwin, tick
    )
    cnt_y, sum_y, _ = _accumulate_channel(
        y_ch, t0, t1, avg_seconds, nwin, tick
    )

    valid = (cnt_x > 0) & (cnt_y > 0)
    if not np.any(valid):
        tick()
        empty = np.full((1, 1), np.nan)
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), empty, np.zeros((1, 1))

    seg_x = sum_x[valid] / cnt_x[valid]
    seg_y = sum_y[valid] / cnt_y[valid]

    x_edges = _edges_from_spec(x_bins, float(np.min(seg_x)), float(np.max(seg_x)))
    y_edges = _edges_from_spec(y_bins, float(np.min(seg_y)), float(np.max(seg_y)))

    if z_ch is None:
        # Duration map: sum of window durations per bin.
        values = _binned_stat(seg_x, seg_y, dur_x[valid], "sum", x_edges, y_edges)
        counts = _binned_stat(seg_x, seg_y, dur_x[valid], "count", x_edges, y_edges)
    else:
        cnt_z, sum_z, _ = _accumulate_channel(
            z_ch, t0, t1, avg_seconds, nwin, tick
        )
        keep = valid & (cnt_z > 0)
        if not np.any(keep):
            tick()
            empty = np.full(
                (len(x_edges) - 1, len(y_edges) - 1), np.nan
            )
            return x_edges, y_edges, empty, np.zeros_like(empty)
        zx = sum_x[keep] / cnt_x[keep]
        zy = sum_y[keep] / cnt_y[keep]
        seg_z = sum_z[keep] / cnt_z[keep]
        values = _binned_stat(zx, zy, seg_z, statistic, x_edges, y_edges)
        counts = _binned_stat(zx, zy, seg_z, "count", x_edges, y_edges)

    values = np.where(counts > 0, values, np.nan)
    tick()
    return x_edges, y_edges, values, counts
