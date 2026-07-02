"""Actions for PCH module: load channels, preview plots, extract TimeData."""

from collections import defaultdict

import numpy as np
from matplotlib import gridspec

from dac.core.actions import PAB, VAB
from . import TimeSegment, TimeChannel
from .loader import TDMSLoader, CSVLoader, HDF5Loader
from .plots import is_datetime_type, setup_datetime_axis
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
                segments = loader.load_meta(fpath, t0=t0, dt=dt)
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
    drawn on the same axes. When *t0* is ``np.datetime64`` the x-axis
    is formatted as date/time; for float *t0* it is labelled "Time [s]".
    """

    CAPTION = "Preview channels"

    def __call__(
        self,
        channels: list[TimeChannel],
        target_fs: float = 1.0,
        t_start=None,
        t_end=None,
    ):
        fig = self.figure
        fig.suptitle("Channel preview")

        ax = fig.gca()
        _datetime_configured = False

        for ch in channels:
            t, y, _ = ch.get_merged_data(
                t_start=t_start, t_end=t_end, target_fs=target_fs
            )
            if len(t) == 0:
                continue

            if not _datetime_configured and is_datetime_type(t):
                setup_datetime_axis(ax)
                _datetime_configured = True

            ax.plot(
                t,
                y,
                label=f"{ch.name} [{ch.y_unit}]",
                linewidth=0.5,
            )

        if not _datetime_configured:
            ax.set_xlabel("Time [s]")

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
        t_start,
        t_end,
    ) -> TimeData:
        t, y, dt = channel.get_merged_data(t_start=t_start, t_end=t_end)
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

        self.message(f"Extracted {len(y)} samples ({len(y) * dt:.2f}s)")
        return TimeData(
            name=f"{channel.name}-Extract",
            y=y,
            dt=dt,
            y_unit=channel.y_unit,
        )


# ---------------------------------------------------------------------------
# Interactive time-range selection
# ---------------------------------------------------------------------------


class SelectTimeRangeAction(VAB):
    """Interactive time-range selection on TimeChannel preview.

    Groups channels by *y_unit* into subplots sharing an x-axis.
    **Left-drag** selects a time span (range mode);
    **left-click** selects a single point (full-file mode).

    After selection right-click the action in the list and choose
    "Setup Analysis Context" to create a new analysis context with
    a ``LoadAndCropAction`` ready to extract data.
    """

    CAPTION = "Select time range for analysis"

    def __call__(
        self,
        channels: list[TimeChannel],
        target_fs: float = 1.0,
    ):
        if not channels:
            return

        self._channels = channels
        self._t_start = None
        self._t_end = None
        self._span_patches = []
        self._vlines = []
        self._axes = []

        fig = self.figure
        fig.suptitle(
            "Drag to select time range  |  Click for a point  |  "
            "Right-click action → Setup Analysis Context"
        )

        # --- group channels by y_unit ---
        unit_groups: list[tuple[str, list[TimeChannel]]] = []
        seen_units: dict[str, int] = {}
        for ch in channels:
            idx = seen_units.get(ch.y_unit)
            if idx is None:
                idx = len(unit_groups)
                unit_groups.append((ch.y_unit, []))
                seen_units[ch.y_unit] = idx
            unit_groups[idx][1].append(ch)

        n_rows = len(unit_groups)
        gs = gridspec.GridSpec(n_rows, 1, figure=fig)
        gs.update(hspace=0.05)
        axes = []
        datetime_setup = False

        self._all_times = []
        all_y_flat = []

        for i, (unit, grp) in enumerate(unit_groups):
            ax = fig.add_subplot(gs[i], sharex=axes[0] if axes else None)
            axes.append(ax)

            for ch in grp:
                t, y, _ = ch.get_merged_data(target_fs=target_fs)
                if len(t) == 0:
                    continue

                if not datetime_setup and is_datetime_type(t):
                    setup_datetime_axis(ax)
                    datetime_setup = True

                ax.plot(t, y, label=f"{ch.name}", linewidth=0.5)
                self._all_times.append(t)
                all_y_flat.extend(y if len(y) else [0])

            ax.set_ylabel(f"[{unit}]")
            ax.legend(loc="upper right", fontsize="small")

        if not datetime_setup and axes:
            axes[-1].set_xlabel("Time [s]")

        self._axes = axes
        self._y_min = float(np.min(all_y_flat)) if all_y_flat else 0
        self._y_max = float(np.max(all_y_flat)) if all_y_flat else 0
        self._y_span = self._y_max - self._y_min or 1.0

        canvas = self.canvas

        # --- event handlers ---
        def _ax_for(event):
            if event.inaxes is not None and event.inaxes in axes:
                return event.inaxes
            return None

        def on_press(event):
            if canvas.widgetlock.locked():
                return
            ax = _ax_for(event)
            if ax is None:
                return
            if event.button == 1:  # left button
                self._t_start = event.xdata
                _clear_spans()
                _clear_vlines()

                if event.dblclick:
                    return
                # span selection
                for ax_i in axes:
                    span = ax_i.axvspan(
                        event.xdata, event.xdata,
                        alpha=0.2, color="green",
                    )
                    self._span_patches.append(span)
                canvas.draw_idle()

        def on_motion(event):
            if self._t_start is None or not self._span_patches:
                return
            x = event.xdata
            if x is None:
                return
            for span in self._span_patches:
                xy = span.get_xy()
                xy[1, 0] = x
                xy[2, 0] = x
                span.set_xy(xy)
            canvas.draw_idle()

        def on_release(event):
            if self._t_start is None:
                return
            _clear_spans()
            x = event.xdata
            if x is None:
                self._t_start = None
                return
            if abs(x - self._t_start) < 1e-12:
                # point mode: draw vertical lines
                self._t_end = self._t_start
                for ax_i in axes:
                    vline = ax_i.axvline(
                        self._t_start, color="red", linestyle="--", linewidth=1
                    )
                    self._vlines.append(vline)
            else:
                # range mode: draw span
                self._t_end = x
                t0, t1 = sorted([self._t_start, self._t_end])
                self._t_start, self._t_end = t0, t1
                for ax_i in axes:
                    span = ax_i.axvspan(
                        t0, t1, alpha=0.15, color="green"
                    )
                    self._span_patches.append(span)
            canvas.draw_idle()
            self.message(
                f"Selected: {self._t_start} → {self._t_end}"
            )

        def _clear_spans():
            for s in self._span_patches:
                s.remove()
            self._span_patches.clear()

        def _clear_vlines():
            for v in self._vlines:
                v.remove()
            self._vlines.clear()

        self._cids.append(canvas.mpl_connect("button_press_event", on_press))
        self._cids.append(canvas.mpl_connect("motion_notify_event", on_motion))
        self._cids.append(canvas.mpl_connect("button_release_event", on_release))


# ---------------------------------------------------------------------------
# Load and crop (standalone, no TimeChannel dependency)
# ---------------------------------------------------------------------------


class LoadAndCropAction(PAB):
    """Load file data and optionally crop to a time range.

    Detects file format by extension, loads metadata via the appropriate
    Loader, then reads full data for channels that overlap ``[t_start, t_end]``
    (or all channels when *t_start* is None or equals *t_end*).
    Returns ``list[TimeData]`` ready for analysis (FFT, filtering, …).

    All loaded arrays go through the process-wide Cache — already-loaded
    data from a previous preview is reused instantly.
    """

    CAPTION = "Load and crop data"

    def __call__(
        self,
        fpaths: list[str],
        t_start=None,
        t_end=None,
    ) -> list[TimeData]:
        results: list[TimeData] = []
        n = len(fpaths)
        is_range = t_start is not None and t_end is not None and t_start < t_end

        for i_file, fpath in enumerate(fpaths):
            lt = _detect_loader_type(fpath)
            loader_cls = _LOADER_MAP.get(lt)
            if loader_cls is None:
                self.message(f"Unknown format: {fpath}")
                continue

            loader = loader_cls()
            segments = loader.load_meta(fpath)

            for seg in segments:
                if seg.t0 is np.datetime64("NaT"):
                    continue

                seg_end = seg.t_end
                if is_range:
                    if seg_end <= t_start or seg.t0 >= t_end:
                        continue
                elif t_start is not None:
                    if seg.t0 > t_start or seg_end < t_start:
                        continue

                data = seg.y  # triggers lazy-load via Cache
                t_axis = seg.t

                if is_range:
                    mask = (t_axis >= t_start) & (t_axis <= t_end)
                    data = data[mask]

                if len(data) == 0:
                    continue

                name = seg.name or f"ch_{i_file}"
                td = TimeData(
                    name=name,
                    y=data,
                    dt=seg.dt,
                    y_unit=seg.y_unit,
                    comment=f"from {fpath}",
                )
                results.append(td)

            self.progress(i_file + 1, n)

        self.message(f"Loaded {len(results)} channels")
        return results
