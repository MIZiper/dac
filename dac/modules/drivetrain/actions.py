"""Actions related to drivetrain components for the DAC framework.

This module provides actions for creating bearings and gearboxes, generating
order lists for gearboxes, and visualizing frequency lines on time-domain
and frequency-domain plots.
"""

import numpy as np
from . import BallBearing, GearboxDefinition, BearingInputStage
from dac.modules.timedata import TimeData
from dac.core.actions import VAB, PAB, SAB, TAB, ActionBase
from dac.modules.timedata.actions import ShowTimeDataAction
from dac.modules.nvh.data import OrderList, OrderInfo
from dac.modules.nvh.actions import ViewFreqDomainAction
from matplotlib.backend_bases import MouseButton, MouseEvent

class CreateBearing(ActionBase):
    CAPTION = "Make a bearing"
    def __call__(self, N_balls: int=8, D_ball: float=2, D_pitch: float=12, beta: float=15, irr: bool=True) -> BallBearing:
        """Creates a BallBearing data node.

        Parameters
        ----------
        N_balls : int
            Number of balls in the bearing.
        D_ball : float
            Diameter of a single ball.
        D_pitch : float
            Pitch diameter of the bearing.
        beta : float
            Contact angle of the bearing in degrees.
        irr : bool, default True
            Inner race is rotating element.

        Returns
        -------
        BallBearing
            A BallBearing object initialized with the given parameters.
        """

        return BallBearing(
            name="Ball bearing",
            N_balls=N_balls,
            D_ball=D_ball,
            D_pitch=D_pitch,
            beta=beta,
            irr=irr,
        )

class CreateGearboxWithBearings(ActionBase):
    CAPTION = "Make gearbox with bearings"
    def __call__(self, gearbox: GearboxDefinition, bearings: list[tuple[BearingInputStage, BallBearing]]) -> GearboxDefinition:
        """Creates a GearboxDefinition data node with associated bearings.

        Parameters
        ----------
        gearbox : GearboxDefinition
            The base GearboxDefinition object.
        bearings : list of (BearingInputStage, BallBearing)
            The bearing mount info respect to gearbox,
            bearing is always defined to mount on stage input shaft.

        Returns
        -------
        GearboxDefinition
            A new GearboxDefinition object with the specified stages and bearings.
        """

        return GearboxDefinition(
            name="Gearbox with bearings",
            stages=gearbox.stages.copy(),
            bearings=bearings,
        )

class GenerateOutputOfGearbox(ActionBase):
    CAPTION = "Convert metric after gearbox"
    def __call__(self, input: TimeData, gearbox: GearboxDefinition, reduce: bool=False) -> TimeData:
        """Convert values after the transmission of a gearbox
        
        It's just a simple conversion based on total ratio.
        Users have to specify whether values become larger (`reduce=False`) or smaller (`reduce=True`).
        Whether the `gearbox` is a reducer or increaser has no impact on the conversion.
        """

        total_ratio = gearbox.total_ratio
        if (total_ratio > 1 and reduce) or (total_ratio < 1 and not reduce):
            total_ratio = 1 / total_ratio

        return TimeData(f"{input.name}-Convert", y=input.y*total_ratio, dt=input.dt, y_unit=input.y_unit)

class CreateOrdersOfGearbox(ActionBase):
    CAPTION = "Create orders for gearbox"
    def __call__(self, gearbox: GearboxDefinition, ref_output: bool=True, ref2speed: bool=False) -> OrderList:
        """Creates an OrderList for a given GearboxDefinition with reference to input/output shaft.

        The input or output shaft will have an order=1, and calculate other predefined frequencies' orders.

        Parameters
        ----------
        gearbox : GearboxDefinition
            The GearboxDefinition to generate orders for.
        ref_output : bool, default True
            If True, the reference is considered to be on the
            output shaft; otherwise, it's on the input shaft.
        ref2speed : bool, default False
            If True, the `order.value` is reference to speed, which is used for order slice with y-axis (ref_bins) in rpm.
            If False, the `order.value` is reference to frequency.

        Returns
        -------
        OrderList
            An OrderList containing OrderInfo objects for the gearbox.
        """

        ol = OrderList(f"Orders-{gearbox.name}")

        speed = 1 if ref2speed else 60

        if ref_output:
            ispeed = speed / gearbox.total_ratio
        else:
            ispeed = speed

        for freq, label in gearbox.get_freqs_labels_at(input_speed=ispeed):
            # when speed==60, the freq value is the order value
            ol.orders.append(OrderInfo(label, freq))

        return ol

class ViewAllOrders(TAB): # should this under dac.modules.nvh.actions?
    CAPTION = "Show drivetrain orders"
    def __call__(self, ol: OrderList, speed: float=None):
        N = 3
        rows = []
        data = []
        cols = [f"{i+1}x" for i in range(N)]

        for o in ol.orders:
            rows.append(o.name)
            row = []
            for c in range(N):
                if speed is None:
                    row.append(f"{o.value * (c+1):.4f}")
                else:
                    row.append(f"{o.value * (c+1) * speed/60:.2f}")
            data.append(row)

        if speed is None:
            title = 'Orders [-]'
        else:
            title = f'Frequencies [Hz] at {speed}'

        stats = {
            "title": f"({ol.name}) {title}",
            "headers": {
                "row": rows,
                "col": cols,
            },
            "data": data
        }
        self.present(stats)

class ShowFreqLinesTime(VAB):
    CAPTION = "Mark frequency lines on time domain"
    def __call__(self, gearbox: GearboxDefinition, speed_channel: TimeData, speed_on_output: bool=True, stages: list[int]=[1, 2], fmt_lines: list[str]=["{f_1}", "{f_2}-{f_1}"]): # bearings: list[tuple[BallBearing, BearingInputStage]]
        """Marks characteristic frequency lines on a time-domain plot.

        This action allows users to click on a time-domain plot, and lines
        representing characteristic frequencies of the gearbox (and optionally
        custom formatted lines) at that time instant's speed will be drawn.

        Parameters
        ----------
        gearbox : GearboxDefinition
            The GearboxDefinition providing the characteristic frequencies.
        speed_channel : TimeData
            TimeData representing the speed profile. The mean of
            this channel is used as the reference speed.
        speed_on_output : bool, default True
            If True, speed_channel is considered to be the
            output shaft speed.
        stages : list[int], optional
            A list of stage numbers (1-indexed) to display main predefined frequencies.
        fmt_lines : list[str], optional
            A list of strings for custom frequency lines.
            Each string can be a format string using labels from
            `gearbox.get_freqs_labels_at` (e.g., "{f_1}", "{fz_2}-{f_1}")
            or "label,frequency_value" (e.g., "MyFreq, 123.4"),
        """

        if not speed_channel or not gearbox:
            return
        
        if stages is None:
            stages = []
        if fmt_lines is None:
            fmt_lines = []

        canvas = self.canvas
        widgets = [] # it's actually patches

        def on_press(event: MouseEvent):
            if ( (not (ax:=event.inaxes)) or event.button!=MouseButton.LEFT or canvas.widgetlock.locked() ):
                return
            for widget in widgets: # widgets from previous press
                widget.remove()
            widgets.clear()

            bits = 0
            for stage_num in stages:
                bits |= 1<<(stage_num-1)
            moment = event.xdata

            trans = ax.get_xaxis_text1_transform(0)
            speed = np.abs(np.mean(speed_channel.y)) # if isnumber(speed_channel), just assign

            if speed_on_output:
                speed = speed / gearbox.total_ratio

            for freq, label in gearbox.get_freqs_labels_at(speed, choice_bits=bits):
                dt = 1 / freq
                x = moment + dt

                widgets.append( ax.axvline(x, ls="--", lw=1) )
                widgets.append( ax.text(x, 1, label, transform=trans[0]) )

            format_dict = {label: freq for freq, label in gearbox.get_freqs_labels_at(speed)}
            for i, fmt_line in enumerate(fmt_lines):
                label, *freqs = fmt_line.split(",", maxsplit=1)

                if freqs: # freq provided
                    freq = float(freqs[0])
                else:
                    freq = eval(label.format(**format_dict))

                dt = 1 / freq
                x = moment + dt
                                
                widgets.append( ax.axvline(x, ymax=0.95-0.05*(i%2), ls="--", lw=1) )
                widgets.append( ax.text(x, 0.95-0.05*(i%2), label, transform=trans[0]) )

            widgets.append(ax.axvline(event.xdata))
            canvas.draw_idle()

        self._cids.append( canvas.mpl_connect("button_press_event", on_press) )

class ShowFreqLinesFreq(VAB):
    CAPTION = "Mark frequency lines on spectrum"
    def __call__(self, gearbox: GearboxDefinition, speed_channel: TimeData, speed_on_output: bool=True, stages: list[int]=[1, 2], fmt_lines: list[str]=["{f_1}", "{f_2}-{f_1}"]):
        """Marks characteristic frequency lines on a frequency-domain plot (spectrum).

        This action allows users to click on a spectrum.
        - Left-click: Draws lines relative to the clicked frequency (sidebands).
        - Right-click: Draws lines from 0 Hz (absolute frequencies).
        Characteristic frequencies are determined from the gearbox and speed channel.

        Parameters
        ----------
        gearbox : GearboxDefinition
            The GearboxDefinition providing the characteristic frequencies.
        speed_channel : TimeData
            TimeData representing the speed profile. The mean of
            this channel is used as the reference speed.
        speed_on_output : bool, default True
            If True, speed_channel is considered to be the
            output shaft speed.
        stages : list[int], optional
            A list of stage numbers (1-indexed) to display main predefined frequencies.
        fmt_lines : list[str], optional
            A list of strings for custom frequency lines.
            Each string can be a format string using labels from
            `gearbox.get_freqs_labels_at` (e.g., "{f_1}", "{fz_2}-{f_1}")
            or "label,frequency_value" (e.g., "MyFreq, 123.4").
        """

        if not speed_channel or not gearbox:
            return
        
        if stages is None:
            stages = []
        if fmt_lines is None:
            # `fmt_lines`, e.g.
            # {f_2}-{f_1}
            # f_custom, 1.1
            # TODO: f_custom, {f_2}-{f_1} # and usable without `speed_channel` or `gearbox`

            fmt_lines = []
        
        fig = self.figure
        ax = fig.gca()

        canvas = self.canvas
        widgets = [] # it's actually patches

        def plot_lines(start_freq: float, sideband: bool=False):
            """Draws the frequency lines on the spectrum axes.

            Clears previous lines and draws new ones based on the start_freq
            and whether sidebands are requested.

            Parameters
            ----------
            start_freq : float
                The starting frequency from which to draw lines or sidebands.
            sideband : bool, default False
                If True, draws sidebands around `start_freq`. If False, draws
                lines from 0 Hz.
            """

            for widget in widgets: # widgets from previous press
                widget.remove()
            widgets.clear()

            bits = 0
            for stage_num in stages:
                bits |= 1<<(stage_num-1)

            trans = ax.get_xaxis_text1_transform(0)
            speed = np.abs(np.mean(speed_channel.y)) # if isnumber(speed_channel), just assign
            if speed_on_output:
                speed = speed / gearbox.total_ratio

            delta_factors = [1] if not sideband else [1, -1]

            for freq, label in gearbox.get_freqs_labels_at(speed, choice_bits=bits):
                # TODO: based on checkbox
                # if 'fz' in label:
                #     continue
                for factor in delta_factors:
                    widgets.append(ax.axvline(start_freq+freq*factor, ls="--", lw=1))
                    widgets.append(ax.text(start_freq+freq*factor, 1, label, transform=trans[0]))

            format_dict = {label: freq for freq, label in gearbox.get_freqs_labels_at(speed)}
            for i, fmt_line in enumerate(fmt_lines):
                label, *freqs = fmt_line.split(",", maxsplit=1)

                if freqs: # freq provided
                    freq = float(freqs[0])
                else:
                    freq = eval(label.format(**format_dict))

                for factor in delta_factors:
                    widgets.append(ax.axvline(start_freq+freq*factor, ymax=0.95-0.05*(i%2), ls="--", lw=1))
                    widgets.append(ax.text(start_freq+freq*factor, 0.95-0.05*(i%2), label, transform=trans[0]))
            
            if sideband:
                widgets.append(ax.axvline(start_freq))
            canvas.draw_idle()

        def on_press(event: MouseEvent):
            if ( (not (ax:=event.inaxes)) or canvas.widgetlock.locked() ):
                return
            if event.button==MouseButton.LEFT:
                plot_lines(event.xdata, sideband=True)
            elif event.button==MouseButton.RIGHT:
                plot_lines(0, sideband=False)

        plot_lines(0, sideband=False) # init plot

        self._cids.append( canvas.mpl_connect("button_press_event", on_press) )

class ShowSpectrumWithFreqLines(SAB, seq=[ViewFreqDomainAction, ShowFreqLinesFreq]):
    CAPTION = "Show FFT spectrum with freq lines"

class ShowTimeDataWithFreqLines(SAB, seq=[ShowTimeDataAction, ShowFreqLinesTime]):
    CAPTION = "Show time data with freq lines"