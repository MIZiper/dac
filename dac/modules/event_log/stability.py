"""Stable-segment detection for event extraction.

Pure, PyQt-free helpers that find time intervals where a TimeChannel's
rolling standard deviation stays below a tolerance, and intersect such
intervals across several channels.  Times are exchanged as epoch seconds
so channels with different time bases (``float`` vs ``np.datetime64``)
can be combined.
"""

import numpy as np


def _to_epoch_seconds(t) -> float:
    """Convert a scalar time (``datetime64`` or float seconds) to epoch seconds."""
    if isinstance(t, np.datetime64):
        return float(t.astype("datetime64[ns]").astype(np.int64)) / 1e9
    return float(t)


def epoch_to_time(epoch: float, is_datetime: bool):
    """Convert epoch seconds back to ``datetime64`` or ``float``."""
    if is_datetime:
        return np.datetime64(int(round(epoch * 1e9)), "ns")
    return float(epoch)


def rolling_std(y, win_samples: int):
    """Rolling population standard deviation over *win_samples* samples.

    Returns ``(std, window_ok)``.  A window is usable only when it is
    complete (the first ``win_samples - 1`` samples lack history) and
    contains no NaN; otherwise ``std`` is ``NaN`` and ``window_ok`` False.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    std = np.full(n, np.nan)
    window_ok = np.zeros(n, dtype=bool)
    if n == 0 or win_samples <= 0:
        return std, window_ok

    w = int(win_samples)
    if w == 1:
        finite = np.isfinite(y)
        std[finite] = 0.0
        window_ok[:] = finite
        return std, window_ok

    valid = np.isfinite(y)
    y0 = np.where(valid, y, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(y0)))
    csum2 = np.concatenate(([0.0], np.cumsum(y0 * y0)))
    cnan = np.concatenate(([0], np.cumsum(~valid)))

    idx = np.arange(n)
    i0 = np.maximum(0, idx - w + 1)
    i1 = idx + 1
    count = (i1 - i0).astype(np.float64)
    s = csum[i1] - csum[i0]
    s2 = csum2[i1] - csum2[i0]
    nan_count = cnan[i1] - cnan[i0]

    ok = (count == w) & (nan_count == 0)
    mean = s / count
    var = np.clip(s2 / count - mean * mean, 0.0, None)
    std[:] = np.sqrt(var)
    std[~ok] = np.nan
    window_ok[:] = ok
    return std, window_ok


def _runs(mask):
    """Contiguous True runs as inclusive ``(i0, i1)`` index pairs."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, [e - 1 for e in ends]))


def _merge(intervals):
    """Sort and merge touching/overlapping ``(start, end)`` intervals."""
    if not intervals:
        return []
    merged = [list(sorted(intervals)[0])]
    for s, e in sorted(intervals)[1:]:
        if s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def stable_intervals(
    channel,
    tolerance: float,
    win_seconds: float,
    min_duration: float = 0.0,
):
    """Stable ``(start_epoch, end_epoch)`` intervals for *channel*.

    A sample is stable when the rolling standard deviation over
    *win_seconds* is ``<= tolerance``.  Runs shorter than *min_duration*
    seconds are dropped; touching/overlapping intervals are merged.
    Segments are scanned one at a time; gaps between them naturally break
    the intervals.
    """
    intervals = []
    for seg in channel.segments:
        if seg.length <= 0:
            continue
        y = seg.y
        if y is None or len(y) == 0:
            continue
        n = min(seg.length, len(y))
        dt = seg.dt
        if n <= 0 or dt <= 0:
            continue

        win_samples = max(1, int(round(win_seconds / dt)))
        std, ok = rolling_std(np.asarray(y)[:n], win_samples)
        stable = ok & (std <= tolerance)

        t0 = _to_epoch_seconds(seg.t0)
        for i0, i1 in _runs(stable):
            if (i1 - i0 + 1) * dt < min_duration:
                continue
            intervals.append((t0 + i0 * dt, t0 + i1 * dt))
    return _merge(intervals)


def intersect_intervals(interval_lists):
    """Intersection of several interval lists.

    Each input is an iterable of ``(start, end)`` pairs.  The result is the
    time covered by every input; an empty input makes the result empty.
    """
    lists = [_merge(list(iv)) for iv in interval_lists]
    if not lists:
        return []
    result = lists[0]
    for other in lists[1:]:
        result = _intersect_two(result, other)
        if not result:
            return []
    return result


def _intersect_two(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s <= e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out
