from typing import Any
import numpy as np
from collections import namedtuple

from dac.core import DataClassNode
from dac.core.actions import ActionBase
from . import TimeData

SineComponent = namedtuple("SineComponent", ['amp', 'freq', 'phase'])

class SignalConstructAction(ActionBase):
    CAPTION = "Construct signal with sines"
    def __call__(self, components: list[SineComponent], duration: float=10, fs: int=1000) -> TimeData:
        r"""Construct time domain data with sine waves

        Parameters
        ----------
        components: [(amplitude, frequency, phase)]
            list of tuples, each tuple contains basic info of the sine wave

            amplitude: float
            frequency: float, [Hz]
            phase: float, [rad]
        fs: int, [Hz]
            sample rate
        duration: float, [s]
            sample time
        """

        t = np.arange(int(duration * fs)) / fs
        y = np.zeros_like(t)
        
        for amp, freq, phase in components:
            y += amp*np.sin(2*np.pi*freq*t + phase)

        return TimeData(name, y, fs=fs, x_unit="s", comment="Constructed time data")