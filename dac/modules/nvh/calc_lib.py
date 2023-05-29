import numpy as np

def to_decibel(value, ref, by_amplitude=True):
    if by_amplitude: # ref = 20 uPa
        return 20*np.log10(value / ref)
    else: # by_power # ref = 10^-12 W
        return 10*np.log10(value / ref)

def a_weighted(freq: np.array | float) -> np.array | float:
    # dB + diff => dB(A)
    
    A_diff = 20 * np.log10(
        (12194**2 * freq**4) /
        ( (freq**2 + 20.6**2) * (((freq**2 + 107.7**2) * (freq**2 + 737.9**2))**0.5) * (freq**2 + 12194**2) )
    ) + 2
    # 20*log_10(RA(f)) - 20*log_10(RA(1000)) = 20*log_10(RA(f)) - (-2)

    return A_diff