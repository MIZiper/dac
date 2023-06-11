import numpy as np

from dac.core.actions import ActionBase, VAB, PAB, SAB
from dac.modules.timedata import TimeData
from . import WindowType, BandCorrection, BinMethod
from .data import FreqIntermediateData, DataBins

class ToFreqDomainAction(ActionBase):
    CAPTION = "Simple FFT to frequency domain"

class ToFreqIntermediateAction(PAB):
    CAPTION = "FFT to frequency domain with window and reference"

    def __call__(self, channels: list[TimeData],
                 window: WindowType=WindowType.Hanning, corr: BandCorrection=BandCorrection.NarrowBand,
                 resolution: float=0.5, overlap: float=0.75,
                 ref_channel: TimeData=None,
                 ) -> list[FreqIntermediateData]:
        
        freqs = []

        window_funcs = {
            WindowType.Hanning: np.hanning,
            WindowType.Hamming: np.hamming,
        }

        if ref_channel is not None:
            ref_batches = ref_channel.to_bins(df=resolution, overlap=overlap)
            ref_bins_y = np.mean(ref_batches, axis=1)
            ref_bins = DataBins(name=ref_channel.name, y=ref_bins_y, y_unit=ref_channel.y_unit)
        # else:
        #     create a TimeData channel, but don't know the length

        for channel in channels:
            batches = channel.to_bins(df=resolution, overlap=overlap)
            N_batches, batch_N = batches.shape

            if ref_channel is None:
                ref_bins_y = np.arange(N_batches) * 1/resolution * (1-overlap)
                ref_bins = DataBins(name="Time", y=ref_bins_y, y_unit="s")
                ref_bins._method = BinMethod.Min

            batches = batches * window_funcs[window](batch_N)
            batches_fft = np.fft.fft(batches) / batch_N * window.value[corr.value]

            double_spec = batches_fft[:, :int(np.ceil(batch_N/2))]
            double_spec[:, 1:] *= 2

            freq = FreqIntermediateData(name=channel.name, z=double_spec, df=resolution, z_unit=channel.y_unit, ref_bins=ref_bins)
            freqs.append(freq)

        return freqs
    
class ViewFreqDomainAction(VAB):
    ...

class ViewFreqIntermediateAction(VAB):
    CAPTION = "Show FFT color plot"

    def __call__(self, channel: FreqIntermediateData, range: tuple[float, float]=None):
        fig = self.figure
        ax = fig.gca()

        fig.suptitle(f"Color map: {channel.name}")
        xs = channel.x
        ax.set_xlabel("Frequency [Hz]")
        if (ref_bins:=channel.ref_bins) is not None:
            ys = channel.ref_bins.y
            ax.set_ylabel(f"{ref_bins.name} [{ref_bins.y_unit}]")
        m = ax.pcolormesh(xs, ys, np.abs(channel.z), cmap='jet')
        cb = fig.colorbar(m)
        cb.set_label(f"Amplitude [{channel.z_unit}]")
        if range is not None:
            ax.set_xlim(range)

# extract specific frequencies
# calc rms
# extract order slice