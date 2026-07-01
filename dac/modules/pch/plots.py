"""Plot utilities for PCH module.

Provides helpers for downsampling, datetime axis setup, and multi-axis
layout for TimeChannel visualization.
"""

import numpy as np
from datetime import datetime, timezone
import matplotlib.dates as mdates


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
        Time axis (absolute epoch seconds).
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

    src_fs = 1.0 / np.mean(np.diff(t))
    interval = int(round(src_fs / target_fs))
    if interval <= 1:
        return t, y, 1.0 / src_fs

    return t[::interval], y[::interval], 1.0 / target_fs


def epoch_to_mpl(t_epoch: np.ndarray) -> np.ndarray:
    """Convert epoch seconds to matplotlib date numbers."""
    return mdates.epoch2num(t_epoch)


def setup_datetime_axis(ax, tz=None):
    """Configure a matplotlib axis for datetime display.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to configure.
    tz : datetime.timezone, optional
        Timezone for display. Defaults to UTC.
    """
    if tz is None:
        tz = timezone.utc
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d %H:%M:%S", tz=tz)
    )
    ax.figure.autofmt_xdate()
