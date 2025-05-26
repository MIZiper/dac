"""Actions related to drivetrain components for the DAC framework.

This module provides actions for creating bearings and gearboxes, generating
order lists for gearboxes, and visualizing frequency lines on time-domain
and frequency-domain plots.
"""
import numpy as np
from . import BallBearing, GearboxDefinition, BearingInputStage
from dac.modules.timedata import TimeData
from dac.core.actions import VAB, PAB, SAB, ActionBase
from dac.modules.timedata.actions import ShowTimeDataAction
from dac.modules.nvh.data import OrderList, OrderInfo
from dac.modules.nvh.actions import ViewFreqDomainAction
from matplotlib.backend_bases import MouseButton, MouseEvent

class CreateBearing(ActionBase):
    CAPTION = "Make a bearing"
    def __call__(self, N_balls: int=8, D_ball: float=2, D_pitch: float=12, beta: float=15) -> BallBearing:
        """Creates a BallBearing data node.

        Parameters
        ----------
        N_balls : int, optional
            Number of balls in the bearing, by default 8.
        D_ball : float, optional
            Diameter of a single ball, by default 2.
        D_pitch : float, optional
            Pitch diameter of the bearing, by default 12.
        beta : float, optional
            Contact angle of the bearing in degrees, by default 15.

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
        )

class CreateGearboxWithBearings(ActionBase):
    CAPTION = "Make gearbox with bearings"
    def __call__(self, gearbox: GearboxDefinition, bearings: list[tuple[BearingInputStage, BallBearing]]) -> GearboxDefinition:
        """Creates a GearboxDefinition data node with associated bearings.

        Parameters
        ----------
        gearbox : GearboxDefinition
            The base GearboxDefinition object.
        bearings : list[tuple[BearingInputStage, BallBearing]]
            A list of tuples, where each tuple contains a
            BearingInputStage and a BallBearing object.

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


class CreateOrdersOfGearbox(ActionBase):
    CAPTION = "Create orders for gearbox"
    def __call__(self, gearbox: GearboxDefinition, ref_output: bool=True) -> OrderList:
        """Creates an OrderList for a given GearboxDefinition.

        Calculates characteristic frequencies of the gearbox at a reference speed
        of 60 (rpm or Hz depending on how speed is interpreted by `get_freqs_labels_at`)
        and creates OrderInfo objects for each.

        Parameters
        ----------
        gearbox : GearboxDefinition
            The GearboxDefinition to generate orders for.
        ref_output : bool, optional
            If True, the reference speed is considered to be on the
            output shaft; otherwise, it's on the input shaft, by default True.

        Returns
        -------
        OrderList
            An OrderList containing OrderInfo objects for the gearbox.
        """
        ol = OrderList(f"Orders-{gearbox.name}")
        for freq, label in gearbox.get_freqs_labels_at(speed=60, speed_on_output=ref_output):
            # reference shaft has order 1
            ol.orders.append(OrderInfo(label, freq/60, freq))

        return ol

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
        speed_on_output : bool, optional
            If True, speed_channel is considered to be the
            output shaft speed, by default True.
        stages : list[int], optional
            A list of stage numbers (1-indexed) to display frequencies for,
            by default [1, 2].
        fmt_lines : list[str], optional
            A list of strings for custom frequency lines.
            Each string can be a format string using labels from
            `gearbox.get_freqs_labels_at` (e.g., "{f_1}", "{fz_2}-{f_1}")
            or "label,frequency_value" (e.g., "MyFreq,123.4"),
            by default ["{f_1}", "{f_2}-{f_1}"].
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
            """Handles mouse press events on the plot.

            Calculates and draws frequency lines based on the clicked point's
            x-coordinate (time) and the gearbox/speed parameters.

            Parameters
            ----------
            event : MouseEvent
                The Matplotlib mouse event.
            """
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

            for freq, label in gearbox.get_freqs_labels_at(speed, speed_on_output, choice_bits=bits):
                dt = 1 / freq
                x = moment + dt

                widgets.append( ax.axvline(x, ls="--", lw=1) )
                widgets.append( ax.text(x, 1, label, transform=trans[0]) )

            format_dict = {label: freq for freq, label in gearbox.get_freqs_labels_at(speed, speed_on_output)}
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
        speed_on_output : bool, optional
            If True, speed_channel is considered to be the
            output shaft speed, by default True.
        stages : list[int], optional
            A list of stage numbers (1-indexed) to display frequencies for,
            by default [1, 2].
        fmt_lines : list[str], optional
            A list of strings for custom frequency lines.
            Each string can be a format string using labels from
            `gearbox.get_freqs_labels_at` (e.g., "{f_1}", "{fz_2}-{f_1}")
            or "label,frequency_value" (e.g., "MyFreq,123.4"),
            by default ["{f_1}", "{f_2}-{f_1}"].
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
            sideband : bool, optional
                If True, draws sidebands around start_freq. If False, draws
                lines from 0 Hz, by default False.
            """
            for widget in widgets: # widgets from previous press
                widget.remove()
            widgets.clear()

            bits = 0
            for stage_num in stages:
                bits |= 1<<(stage_num-1)

            trans = ax.get_xaxis_text1_transform(0)
            speed = np.abs(np.mean(speed_channel.y)) # if isnumber(speed_channel), just assign

            delta_factors = [1] if not sideband else [1, -1]

            for freq, label in gearbox.get_freqs_labels_at(speed, speed_on_output, choice_bits=bits):
                # TODO: based on checkbox
                # if 'fz' in label:
                #     continue
                for factor in delta_factors:
                    widgets.append(ax.axvline(start_freq+freq*factor, ls="--", lw=1))
                    widgets.append(ax.text(start_freq+freq*factor, 1, label, transform=trans[0]))

            format_dict = {label: freq for freq, label in gearbox.get_freqs_labels_at(speed, speed_on_output)}
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
            """Handles mouse press events on the spectrum plot.

            Determines whether to plot absolute frequencies or sidebands based
            on the mouse button pressed.

            Parameters
            ----------
            event : MouseEvent
                The Matplotlib mouse event.
            """
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