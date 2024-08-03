from enum import Enum
import numpy as np

class WindowType(Enum):
    Uniform = (1.0, 1.0)
    Hanning = (2.0, 1.63)
    Hamming = (1.85, 1.59)
    KaiserBessel = (2.49, 1.86)
    FlatTop = (4.18, 2.26)
    Blackman = (2.80, 1.97)
    Unknown = (1.0, 1.0) # (BandCorrection.NarrowBand, BandCorrection.BroadBand)

class BandCorrection(Enum):
    NarrowBand = 0 # amplitude correction
    BroadBand = 1 # power correction

class AmplitudeType(Enum):
    RMS = 1/np.sqrt(2)
    Peak = 1

class AverageType(Enum):
    Energy = 0
    Linear = 1
    EnergyExponential = 2
    Maximum = 3
    Minimum = 4

class FilterType(Enum):
    LowPass = "lowpass"
    HighPass = "highpass"
    BandPass = "bandpass"
    BandStop = "bandstop"

class BinMethod(Enum):
    Mean = 0
    Min = 1

class ToleranceType(Enum):
    FixLines = 0
    FixBandWidth = 1
    FixOrderWidth = 2
    FixFrequencyPercentage = 3