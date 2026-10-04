"""Live interactions for the drivetrain module.

Provides switchable characteristic-frequency marking on time-domain and
frequency-domain plots, derived from
:class:`~dac.modules.drivetrain.ShowFreqLinesTime` /
``ShowFreqLinesFreq`` but as toggleable tools instead of standalone plots.

These tools use only Matplotlib, so they work in the PyQt GUI and in the
web (webagg) front-end alike.
"""

import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent

from dac.core.interact import ToolBase
from dac.modules.timedata import TimeData
from dac.modules.drivetrain import GearboxDefinition


class _FreqLinesBase(ToolBase):
    _DEFAULT_STAGES = [1, 2]
    _DEFAULT_FMT_LINES = ["{f_1}", "{f_2}-{f_1}"]

    def __init__(
        self,
        ctx,
        gearbox: GearboxDefinition = None,
        speed_channel: TimeData = None,
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

    @classmethod
    def available(cls, ctx) -> bool:
        return (
            super().available(ctx)
            and bool(ctx.find_nodes(GearboxDefinition))
            and bool(ctx.find_nodes(TimeData))
        )

    def param_options(self, name: str) -> list:
        if name == "gearbox":
            return list(self.ctx.find_nodes(GearboxDefinition))
        if name == "speed_channel":
            return list(self.ctx.find_nodes(TimeData))
        return []

    def attach(self) -> None:
        self.auto_bind()
        self._lines = []
        if self.stages is None:
            self.stages = list(self._DEFAULT_STAGES)
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

    def _speed(self) -> float:
        speed = np.abs(np.mean(self.speed_channel.y))
        if self.speed_on_output:
            speed = speed / self.gearbox.total_ratio
        return float(speed)

    def _bits(self) -> int:
        bits = 0
        for stage_num in self.stages:
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
    """Click a time-domain plot to mark characteristic frequency lines."""

    CAPTION = "Frequency lines"
    _HINT = "Click on the time plot to mark characteristic frequency lines"

    def on_press(self, event: MouseEvent) -> None:
        canvas = self.ctx.figure.canvas
        gearbox, speed_channel = self.gearbox, self.speed_channel
        if gearbox is None or speed_channel is None:
            return
        if (
            event.inaxes is None
            or event.button != MouseButton.LEFT
            or canvas.widgetlock.locked()
        ):
            return

        ax = event.inaxes
        self._clear_lines()

        moment = event.xdata
        trans = ax.get_xaxis_text1_transform(0)
        speed = self._speed()
        bits = self._bits()

        for freq, label in gearbox.get_freqs_labels_at(speed, choice_bits=bits):
            if not freq:
                continue
            x = moment + 1 / freq
            self._lines.append(ax.axvline(x, ls="--", lw=1))
            self._lines.append(ax.text(x, 1, label, transform=trans[0]))

        format_dict = {
            label: freq for freq, label in gearbox.get_freqs_labels_at(speed)
        }
        for i, fmt_line in enumerate(self.fmt_lines):
            label, freq = self._format_freq(fmt_line, format_dict)
            if not freq:
                continue
            x = moment + 1 / freq
            ypos = 0.95 - 0.05 * (i % 2)
            self._lines.append(ax.axvline(x, ymax=ypos, ls="--", lw=1))
            self._lines.append(ax.text(x, ypos, label, transform=trans[0]))

        self._lines.append(ax.axvline(event.xdata))
        self.redraw()


class FreqLinesSpectrumTool(_FreqLinesBase):
    """Click a spectrum to mark characteristic frequency lines/sidebands."""

    CAPTION = "Frequency lines"
    _HINT = "Left-click = sidebands  |  Right-click = absolute frequencies"

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

        trans = ax.get_xaxis_text1_transform(0)
        speed = self._speed()
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
