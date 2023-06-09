import numpy as np

from dac.core.actions import ActionBase
from dac.modules.timedata import TimeData
from . import WindowType, FreqAlongData

class ToFreqDomainAction(ActionBase):
    CAPTION = "FFT to frequency domain"

    def __call__(self, channels: list[TimeData], window: WindowType, resolution: float=0.5, overlap: float=0.75):
        df = resolution
        freqs = []

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

            freq = FreqAlongData(double_spec)

# extract specific frequencies
# calc rms
# extract order slice