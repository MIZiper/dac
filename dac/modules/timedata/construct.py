"""Provides an action for constructing time-series data from cosine components.

A constructor when no real data available, and serves for functionality validation.
"""

import numpy as np
from collections import namedtuple

from dac.core.actions import ActionBase
from . import TimeData

CosineComponent = namedtuple("CosineComponent", ['freq', 'amp', 'phase'])

class SignalConstructAction(ActionBase):
    CAPTION = "Construct signal with cosines"
    def __call__(self, components: list[CosineComponent], offset: float=0, duration: float=10, fs: int=1000) -> TimeData:
        """Constructs time-domain data from a sum of cosine waves.

        Parameters
        ----------
        components : list[CosineComponent]
            A list of CosineComponent namedtuples, where each namedtuple
            (freq, amp, phase) defines a cosine wave.
            - freq (float): Frequency of the cosine wave in Hz.
            - amp (float): Amplitude of the cosine wave.
            - phase (float): Phase of the cosine wave in degrees.
        offset : float
            A float representing the DC offset of the signal.
        duration : float
            The total duration of the signal in seconds.
        fs : number
            The sampling frequency in Hz.

        Returns
        -------
        TimeData
            A TimeData object representing the generated signal.
        """
        
        t = np.arange(int(duration * fs)) / fs
        y = np.zeros_like(t) + offset
        
        for freq, amp, phase in components:
            y += amp*np.cos(2*np.pi*freq*t + np.deg2rad(phase))

        return TimeData(name="Generated signal", y=y, dt=1/fs, y_unit="-", comment="Constructed time data")