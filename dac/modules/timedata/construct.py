"""Provides an action for constructing time-series data from cosine components.

This module defines `SignalConstructAction`, which allows users to generate
`TimeData` objects by specifying a list of cosine wave components (frequency,
amplitude, phase), an offset, duration, and sampling frequency.
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
        offset : float, optional
            A float representing the DC offset of the signal, by default 0.
        duration : float, optional
            The total duration of the signal in seconds, by default 10.
        fs : int, optional
            The sampling frequency in Hz, by default 1000.

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