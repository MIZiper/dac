import numpy as np
from . import BallBearing, GearboxDefinition, BearingInputStage
from dac.modules.timedata import TimeData
from dac.core.actions import VAB, PAB, SAB, ActionBase
from dac.modules.timedata.actions import ShowTimeDataAction
from dac.modules.nvh.data import OrderList, OrderInfo
from dac.modules.nvh.actions import ViewFreqDomainAction

class CreateBearing(ActionBase):
    CAPTION = "Make a bearing"
    def __call__(self, N_balls: int=8, D_ball: float=2, D_pitch: float=12, beta: float=15) -> BallBearing:
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
        return GearboxDefinition(
            name="Gearbox with bearings",
            stages=gearbox.stages.copy(),
            bearings=bearings,
        )


class CreateOrdersOfGearbox(ActionBase):
    CAPTION = "Create orders for gearbox"
    def __call__(self, gearbox: GearboxDefinition, ref_output: bool=True) -> OrderList:
        ol = OrderList(f"Orders-{gearbox.name}")
        for freq, label in gearbox.get_freqs_labels_at(speed=60, speed_on_output=ref_output):
            # reference shaft has order 1
            ol.orders.append(OrderInfo(label, freq/60, freq))

        return ol

class ShowFreqLinesTime(VAB):
    CAPTION = "Mark frequency lines on time domain"
    def __call__(self, gearbox: GearboxDefinition, speed_channel: TimeData, speed_on_output: bool=True, stages: list[int]=[1, 2], fmt_lines: list[str]=["{f_1}", "{f_2}-{f_1}"]): # bearings: list[tuple[BallBearing, BearingInputStage]]
        if not speed_channel or not gearbox:
            return
        
        if stages is None:
            stages = []
        if fmt_lines is None:
            fmt_lines = []

        canvas = self.canvas
        widgets = [] # it's actually patches

        def on_press(event):
            if ( (not (ax:=event.inaxes)) or event.button!=1 or canvas.widgetlock.locked() ):
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

        bits = 0
        for stage_num in stages:
            bits |= 1<<(stage_num-1)

        trans = ax.get_xaxis_text1_transform(0)
        speed = np.abs(np.mean(speed_channel.y)) # if isnumber(speed_channel), just assign

        for freq, label in gearbox.get_freqs_labels_at(speed, speed_on_output, choice_bits=bits):
            # TODO: based on checkbox
            # if 'fz' in label:
            #     continue

            ax.axvline(freq, ls="--", lw=1)
            ax.text(freq, 1, label, transform=trans[0])

        format_dict = {label: freq for freq, label in gearbox.get_freqs_labels_at(speed, speed_on_output)}
        for i, fmt_line in enumerate(fmt_lines):
            label, *freqs = fmt_line.split(",", maxsplit=1)

            if freqs: # freq provided
                freq = float(freqs[0])
            else:
                freq = eval(label.format(**format_dict))

            ax.axvline(freq, ymax=0.95-0.05*(i%2), ls="--", lw=1)
            ax.text(freq, 0.95-0.05*(i%2), label, transform=trans[0])

class ShowSpectrumWithFreqLines(SAB, seq=[ViewFreqDomainAction, ShowFreqLinesFreq]):
    CAPTION = "Show FFT spectrum with freq lines"

class ShowTimeDataWithFreqLines(SAB, seq=[ShowTimeDataAction, ShowFreqLinesTime]):
    CAPTION = "Show time data with freq lines"