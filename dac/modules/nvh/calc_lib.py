"""Provides calculation utilities for NVH (Noise, Vibration, and Harshness) analysis.
"""

import numpy as np

def to_decibel(value, ref, by_amplitude=True):
    """Converts a given value to decibels (dB) relative to a reference value.

    Parameters
    ----------
    value : array_like
        The input value (or array of values).
    ref : float
        The reference value.
    by_amplitude : bool
        If True (default), calculates dB based on amplitude ratio
        (20*log10). If False, calculates based on power ratio
        (10*log10).

    Returns
    -------
    ndarray or float
        The decibel value(s).
    """

    if by_amplitude: # ref = 20 uPa
        return 20*np.log10(value / ref)
    else: # by_power # ref = 10^-12 W
        return 10*np.log10(value / ref)

def a_weighted(freq: np.array | float) -> np.array | float:
    """Calculates the A-weighting correction factor in dB.

    A-weighting is commonly used in acoustics to adjust sound pressure levels
    to reflect the human ear's sensitivity at different frequencies.

    Parameters
    ----------
    freq : array_like or float
        The frequency (or frequencies) in Hz for which to calculate
        the A-weighting.

    Returns
    -------
    ndarray or float
        The A-weighting correction(s) in dB. This value should be
        added to the original dB level.
    """
    
    # dB + diff => dB(A)
    
    A_diff = 20 * np.log10(
        (12194**2 * freq**4) /
        ( (freq**2 + 20.6**2) * (((freq**2 + 107.7**2) * (freq**2 + 737.9**2))**0.5) * (freq**2 + 12194**2) )
    ) + 2
    # 20*log_10(RA(f)) - 20*log_10(RA(1000)) = 20*log_10(RA(f)) - (-2)

    return A_diff