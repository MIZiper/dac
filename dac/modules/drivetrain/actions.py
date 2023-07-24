from dac.core.actions import VAB, PAB, SAB
from . import BallBearing, GearboxDefinition
from dac.modules.timedata import TimeData
import numpy as np

class BearingInputStage(int): # cannot use namedtuple(BallBearing, BearingInputStage) because no conversion for namedtuple
    pass

class ShowFreqLinesTime(VAB):
    CAPTION = "Mark frequency lines on time domain"
    def __call__(self, bearings: list[tuple[BallBearing, BearingInputStage]]):
        ...

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

            fmt_lines = []
        
        fig = self.figure
        ax = fig.gca()

        bits = 0
        for stage_num in stages:
            bits |= 1<<(stage_num-1)

        trans = ax.get_xaxis_text1_transform(0)
        speed = np.mean(speed_channel.y) # if isnumber(speed_channel), just assign

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