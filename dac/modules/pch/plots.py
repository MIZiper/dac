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
