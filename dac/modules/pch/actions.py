"""Actions for PCH module: load channels, preview plots, extract TimeData."""

from collections import defaultdict

import matplotlib.dates as mdates
import numpy as np

from dac.core.actions import PAB, VAB
from . import TimeSegment, TimeChannel
from .loader import TDMSLoader, CSVLoader, HDF5Loader
from .plots import downsample_array, downsample_time_data, setup_datetime_axis
from dac.modules.timedata import TimeData


_LOADER_MAP = {
    "tdms": TDMSLoader,
    "csv": CSVLoader,
    "hdf5": HDF5Loader,
    "h5": HDF5Loader,
}


def _detect_loader_type(fpath: str) -> str:
    ext = fpath.rsplit(".", 1)[-1].lower() if "." in fpath else ""
    if ext in ("tdms",):
        return "tdms"
    if ext in ("csv",):
        return "csv"
    if ext in ("h5", "hdf5"):
        return "hdf5"
    return "tdms"


def _group_by_channel_name(
    segments: list[TimeSegment],
) -> list[TimeChannel]:
    groups = defaultdict(list)
    for seg in segments:
        name = seg.name or "unknown"
        groups[name].append(seg)

    channels = []
    for ch_name, segs in groups.items():
        ch = TimeChannel(name=ch_name)
        for seg in segs:
            ch.add_segment(seg)
        channels.append(ch)
    return channels


class LoadChannelAction(PAB):
    """Load acquisition files and group segments into TimeChannels.

    Detects file format by extension. Use ``loader_type`` to override.
    TDMS files use their built-in time metadata; CSV uses *t0* and *dt*
    parameters.
    """

    CAPTION = "Load acquisition channels"

    def __call__(
        self,
        fpaths: list[str],
        loader_type: str = "",
        t0: float = 0.0,
        dt: float = 1.0,
    ) -> list[TimeChannel]:
        all_segments: list[TimeSegment] = []
        n = len(fpaths)

        for i, fpath in enumerate(fpaths):
            lt = loader_type or _detect_loader_type(fpath)
            loader_cls = _LOADER_MAP.get(lt)
            if loader_cls is None:
                self.message(f"Unknown loader type '{lt}', skipping {fpath}")
                continue

            loader = loader_cls()
            if lt == "csv":
                segments = loader.load_meta(
                    fpath, t0=t0, dt=dt
                )
            else:
                segments = loader.load_meta(fpath)

            all_segments.extend(segments)
            self.progress(i + 1, n)

        channels = _group_by_channel_name(all_segments)
        self.message(
            f"Loaded {len(all_segments)} segments into {len(channels)} channels"
        )
        return channels


class PreviewChannelAction(VAB):
    """Plot TimeChannels with downsampled preview.

    Data is downsampled to ~1 Hz for fast rendering. Each channel is
    drawn on the same axes with absolute time (datetime-formatted x-axis).
    """

    CAPTION = "Preview channels"

    def __call__(
        self,
        channels: list[TimeChannel],
        target_fs: float = 1.0,
        t_start: float = None,
        t_end: float = None,
    ):
        fig = self.figure
        fig.suptitle("Channel preview")

        ax = fig.gca()
        ax.set_xlabel("Time")
        setup_datetime_axis(ax)

        for ch in channels:
            t, y, _ = ch.get_merged_data(
                t_start=t_start, t_end=t_end, target_fs=target_fs
            )
            if len(t) == 0:
                continue
            t_mpl = mdates.epoch2num(t)
            ax.plot(
                t_mpl,
                y,
                label=f"{ch.name} [{ch.y_unit}]",
                linewidth=0.5,
            )

        ax.legend(loc="upper right")


class ExtractTimeDataAction(PAB):
    """Extract a time range from a TimeChannel as a TimeData node.

    The resulting TimeData can be used for FFT, filtering, and other
    downstream analysis.
    """

    CAPTION = "Extract TimeData from channel"

    def __call__(
        self,
        channel: TimeChannel,
        t_start: float,
        t_end: float,
    ) -> TimeData:
        t, y, dt = channel.get_merged_data(
            t_start=t_start, t_end=t_end
        )
        if len(y) == 0:
            self.message(
                f"No data in range [{t_start}, {t_end}] for {channel.name}"
            )
            return TimeData(
                name=f"{channel.name}-Extract",
                y=np.array([]),
                dt=dt,
                y_unit=channel.y_unit,
            )

        self.message(
            f"Extracted {len(y)} samples ({len(y)*dt:.2f}s)"
        )
        return TimeData(
            name=f"{channel.name}-Extract",
            y=y,
            dt=dt,
            y_unit=channel.y_unit,
        )
