import numpy as np

from dac.core.actions import ActionBase, VAB, PAB, SAB
from dac.modules.timedata import TimeData
from . import WindowType, BandCorrection
from .data import FreqIntermediateData

class ToFreqDomainAction(ActionBase):
    CAPTION = "FFT to frequency domain"

    def __call__(self, channels: list[TimeData],
                 window: WindowType=WindowType.Hanning, corr: BandCorrection=BandCorrection.NarrowBand,
                 resolution: float=0.5, overlap: float=0.75,
                 ref_channel: TimeData=None,
                 ) -> list[FreqIntermediateData]:
        
        df = resolution
        freqs = []

        window_funcs = {
            WindowType.Hanning: np.hanning,
            WindowType.Hamming: np.hamming,
        }

        for channel in channels:
            y = channel.y
            batch_N = np.int( 1/df * channel.fs )
            stride_N = np.int( batch_N * (1-overlap) )
            N_batches = (channel.length-batch_N) // stride_N + 1
            stride, = y.strides
            assert N_batches > 0

            batches = np.lib.stride_tricks.as_strided(y, shape=(N_batches, batch_N), strides=(stride*stride_N, stride))
            # batches -= batches_mean
            # # the code above will cause problem, it's a `as_strided` mapping
            # # corresponding values are connected
            batches = batches * window_funcs[window](batch_N)
            batches_fft = np.fft.fft(batches) / batch_N * window.value[corr.value]

            double_spec = batches_fft[:, :np.int(np.ceil(batch_N/2))]
            double_spec[:, 1:] *= 2

            freq = FreqIntermediateData(name=channel.name, z=double_spec, z_unit=channel.y_unit)
            freqs.append(freq)

        return freqs
    
class ViewFreqDomainAction(VAB):
    ...

class ViewFreqIntermediateAction(VAB):
    ...

# extract specific frequencies
# calc rms
# extract order slice