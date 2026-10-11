"""Live interactions for the drivetrain module.

Provides switchable characteristic-frequency marking on time-domain and
frequency-domain plots, derived from
:class:`~dac.modules.drivetrain.ShowFreqLinesTime` /
``ShowFreqLinesFreq`` but as toggleable tools instead of standalone plots.

These tools use only Matplotlib, so they work in the PyQt GUI and in the
web (webagg) front-end alike.

The time-domain tool reads its reference speed from a pch
``TimeChannel``/``TSChannel`` **near the clicked instant** (not an average
over the whole record), since speed varies during a run.  The spectrum
tool works on steady-state ``TimeData``.
"""

import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent

from dac.core.interact import ToolBase, _axes_have_datetime, _num_to_time
from dac.modules.pch import TimeChannel, TSChannel, TimeSegment
from dac.modules.timedata import TimeData
from dac.modules.drivetrain import GearboxDefinition


def _nearest_in_segment(seg, moment):
    """Nearest non-NaN sample of one segment to *moment* (or ``(None,None)``)."""
    if isinstance(seg, TimeSegment):
        y = seg.y
        n = len(y)
        if n == 0:
            return None, None
        t0, dt = seg.t0, seg.dt
        if isinstance(t0, np.datetime64):
            d = (
                np.datetime64(moment).astype("datetime64[ns]")
                - t0.astype("datetime64[ns]")
            ).astype(np.int64)
            idx = int(round(d / (dt * 1e9)))
            t_val = t0 + np.timedelta64(int(round(idx * dt * 1e9)), "ns")
        else:
            idx = int(round((moment - t0) / dt))
            t_val = t0 + idx * dt
        idx = min(max(idx, 0), n - 1)
        if np.isnan(y[idx]):
            return None, None
        return t_val, y[idx]

    # TSSegment: explicit timestamp array
    t, y = seg.t, seg.y
    if len(t) == 0:
        return None, None
    idx = int(np.searchsorted(t, moment))
    best = None
    for j in (idx - 1, idx, idx + 1):
        if 0 <= j < len(t) and not np.isnan(y[j]):
            dist = abs(t[j] - moment)
            if best is None or dist < best[0]:
                best = (dist, t[j], y[j])
    if best is None:
        return None, None
    return best[1], best[2]


def nearest_sample(channel, moment):
    """Nearest ``(time, value)`` of a pch channel at *moment*.

    Returns ``(None, None)`` when *moment* falls in a gap between segments.
    """
    if moment is None:
        return None, None
    for seg in channel.segments:
        t0, t_end = seg.t0, seg.t_end
        if t0 is None or t_end is None:
            continue
        if t0 <= moment <= t_end:
            return _nearest_in_segment(seg, moment)
    return None, None


class _FreqLinesBase(ToolBase):
    _DEFAULT_FMT_LINES = ["{f_1}", "{f_2}-{f_1}"]

    def __init__(
        self,
        ctx,
        gearbox: GearboxDefinition = None,
        speed_channel=None,
        speed_on_output: bool = True,
        stages: list[int] = None,
        fmt_lines: list[str] = None,
    ) -> None:
        super().__init__(ctx)
        self.gearbox = gearbox
        self.speed_channel = speed_channel
        self.speed_on_output = speed_on_output
        self.stages = stages
        self.fmt_lines = fmt_lines
        self._lines: list = []

    # -- config / availability --------------------------------------------

    def is_ready(self) -> bool:
        """Both a gearbox and a speed reference must be configured.

        Nothing is auto-filled; choose them explicitly in the editor.
        """
        return self.gearbox is not None and self.speed_channel is not None

    def attach(self) -> None:
        self._lines = []
        if self.fmt_lines is None:
            self.fmt_lines = list(self._DEFAULT_FMT_LINES)
        self.notify(self._HINT)

    def detach(self) -> None:
        self._clear_lines()
        super().detach()

    def _clear_lines(self) -> None:
        for artist in self._lines:
            try:
                artist.remove()
            except Exception:
                pass
        self._lines.clear()

    # -- speed + shared drawing helpers -----------------------------------

    def _speed_value(self, moment=None):
        """Raw speed at *moment* (subclass-specific)."""
        raise NotImplementedError

    def _speed_at(self, moment=None):
        """Reference speed on the gearbox output, or ``None`` if unknown."""
        if self.gearbox is None:
            return None
        raw = self._speed_value(moment)
        if raw is None:
            return None
        speed = abs(float(raw))
        if self.speed_on_output:
            speed = speed / self.gearbox.total_ratio
        return speed

    def _bits(self) -> int:
        """Bit mask of selected stages; ``stages: null`` means no stage lines."""
        bits = 0
        for stage_num in self.stages or []:
            bits |= 1 << (stage_num - 1)
        return bits

    def _format_freq(self, fmt_line: str, format_dict: dict):
        label, *freqs = fmt_line.split(",", maxsplit=1)
        if freqs:
            try:
                freq = float(freqs[0])
            except ValueError:
                return None, None
        else:
            try:
                freq = eval(label.format(**format_dict))
            except Exception:
                return None, None
        try:
            return label, float(freq)
        except (TypeError, ValueError):
            return None, None


class FreqLinesTimeTool(_FreqLinesBase):
    """Click a time-domain plot to mark characteristic frequency lines.

    The reference speed is the nearest sample of *speed_channel* (a pch
    ``TimeChannel``/``TSChannel``) at the clicked instant.  Clicking in a
    gap between segments draws nothing.
    """

    CAPTION = "Frequency lines"
    _HINT = "Click on the time plot to mark characteristic frequency lines"

    def __init__(
        self,
        ctx,
        gearbox: GearboxDefinition = None,
        speed_channel: TimeChannel | TSChannel = None,
        speed_on_output: bool = True,
        stages: list[int] = None,
        fmt_lines: list[str] = None,
    ) -> None:
        super().__init__(ctx, gearbox, speed_channel, speed_on_output, stages, fmt_lines)

    @classmethod
    def available(cls, ctx) -> bool:
        return (
            super().available(ctx)
            and bool(ctx.find_nodes(GearboxDefinition))
            and (bool(ctx.find_nodes(TimeChannel)) or bool(ctx.find_nodes(TSChannel)))
        )

    def _speed_value(self, moment=None):
        if moment is None or self.speed_channel is None:
            return None
        _t, value = nearest_sample(self.speed_channel, moment)
        return value

    def on_press(self, event: MouseEvent) -> None:
        canvas = self.ctx.figure.canvas
        if self.gearbox is None or self.speed_channel is None:
            return
        if (
            event.inaxes is None
            or event.button != MouseButton.LEFT
            or event.xdata is None
            or canvas.widgetlock.locked()
        ):
            return

        ax = event.inaxes
        self._clear_lines()

        is_datetime = _axes_have_datetime([ax])
        moment = _num_to_time(event.xdata) if is_datetime else event.xdata
        speed = self._speed_at(moment)
        if not speed:
            # no speed available (gap) or zero speed -> nothing to draw
            self.redraw()
            return

        scale = (1.0 / 86400.0) if is_datetime else 1.0  # seconds -> axis units
        trans = ax.get_xaxis_text1_transform(0)
        bits = self._bits()

        for freq, label in self.gearbox.get_freqs_labels_at(speed, choice_bits=bits):
            if not freq:
                continue
            x = event.xdata + (1 / freq) * scale
            self._lines.append(ax.axvline(x, ls="--", lw=1))
            self._lines.append(ax.text(x, 1, label, transform=trans[0]))

        format_dict = {
            label: freq for freq, label in self.gearbox.get_freqs_labels_at(speed)
        }
        for i, fmt_line in enumerate(self.fmt_lines):
            label, freq = self._format_freq(fmt_line, format_dict)
            if not freq:
                continue
            x = event.xdata + (1 / freq) * scale
            ypos = 0.95 - 0.05 * (i % 2)
            self._lines.append(ax.axvline(x, ymax=ypos, ls="--", lw=1))
            self._lines.append(ax.text(x, ypos, label, transform=trans[0]))

        self._lines.append(ax.axvline(event.xdata))
        self.redraw()


class FreqLinesSpectrumTool(_FreqLinesBase):
    """Click a spectrum to mark characteristic frequency lines/sidebands.

    Uses a steady-state ``TimeData`` channel as the speed reference.
    """

    CAPTION = "Frequency lines"
    _HINT = "Left-click = sidebands  |  Right-click = absolute frequencies"

    def __init__(
        self,
        ctx,
        gearbox: GearboxDefinition = None,
        speed_channel: TimeData = None,
        speed_on_output: bool = True,
        stages: list[int] = None,
        fmt_lines: list[str] = None,
    ) -> None:
        super().__init__(ctx, gearbox, speed_channel, speed_on_output, stages, fmt_lines)

    @classmethod
    def available(cls, ctx) -> bool:
        return (
            super().available(ctx)
            and bool(ctx.find_nodes(GearboxDefinition))
            and bool(ctx.find_nodes(TimeData))
        )

    def _speed_value(self, moment=None):
        if self.speed_channel is None:
            return None
        return float(np.mean(self.speed_channel.y))

    def attach(self) -> None:
        super().attach()
        if self.gearbox is not None and self.speed_channel is not None:
            self._plot_lines(0, sideband=False)

    def _plot_lines(self, start_freq: float, sideband: bool = False) -> None:
        gearbox, speed_channel = self.gearbox, self.speed_channel
        if gearbox is None or speed_channel is None:
            return

        fig = self.ctx.figure
        ax = fig.gca()
        self._clear_lines()

        speed = self._speed_at()
        if not speed:
            self.redraw()
            return

        trans = ax.get_xaxis_text1_transform(0)
        bits = self._bits()
        delta_factors = [1] if not sideband else [1, -1]

        for freq, label in gearbox.get_freqs_labels_at(speed, choice_bits=bits):
            for factor in delta_factors:
                x = start_freq + freq * factor
                self._lines.append(ax.axvline(x, ls="--", lw=1))
                self._lines.append(ax.text(x, 1, label, transform=trans[0]))

        format_dict = {
            label: freq for freq, label in gearbox.get_freqs_labels_at(speed)
        }
        for i, fmt_line in enumerate(self.fmt_lines):
            label, freq = self._format_freq(fmt_line, format_dict)
            if not freq:
                continue
            ypos = 0.95 - 0.05 * (i % 2)
            for factor in delta_factors:
                x = start_freq + freq * factor
                self._lines.append(ax.axvline(x, ymax=ypos, ls="--", lw=1))
                self._lines.append(ax.text(x, ypos, label, transform=trans[0]))

        if sideband:
            self._lines.append(ax.axvline(start_freq))
        self.redraw()

    def on_press(self, event: MouseEvent) -> None:
        canvas = self.ctx.figure.canvas
        if event.inaxes is None or canvas.widgetlock.locked():
            return
        if event.button == MouseButton.LEFT:
            self._plot_lines(event.xdata, sideband=True)
        elif event.button == MouseButton.RIGHT:
            self._plot_lines(0, sideband=False)
