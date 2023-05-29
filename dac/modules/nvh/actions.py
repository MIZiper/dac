from dac.core.actions import ActionBase
from dac.modules.timedata import TimeData
from . import WindowType

class ToFreqDomainAction(ActionBase):
    CAPTION = "FFT to frequency domain"

    def __call__(self, channels: list[TimeData], window: WindowType, resolution: float=0.5, overlap: float=0.75):
        ...

# extract specific frequencies
# calc rms
# extract order slice