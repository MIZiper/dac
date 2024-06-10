import numpy as np
from matplotlib.gridspec import GridSpec

from dac.core.actions import ActionBase, VAB, PAB, SAB
from dac.modules.timedata import TimeData
from . import WindowType, BandCorrection, BinMethod, AverageType
from .data import FreqIntermediateData, DataBins, FreqDomainData

class ToFreqDomainAction(PAB):
    CAPTION = "Simple FFT to frequency domain" # rect window

    def __call__(self, channels: list[TimeData], window: WindowType=WindowType.Uniform, corr: BandCorrection=BandCorrection.NarrowBand) -> list[FreqDomainData]:
        rst = []
        m = len(channels)
        window_funcs = {
            WindowType.Hanning: np.hanning,
            WindowType.Hamming: np.hamming,
        }
        for i, ch in enumerate(channels):
            batch_N = ch.length
            df = 1 / (batch_N*ch.dt)
            
            if window in window_funcs:
                windowed_y = ch.y * window_funcs[window](batch_N)
            else:
                window = WindowType.Uniform
                windowed_y = ch.y
            fdata = np.fft.fft(windowed_y) / batch_N * window.value[corr.value]

            double_spec = fdata[:int(np.ceil(batch_N/2))]
            double_spec[1:] *= 2

            rst.append(FreqDomainData(name=ch.name, y=double_spec, df=df, y_unit=ch.y_unit))
            self.progress(i+1, m)
        return rst

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

        n = len(channels)
        for i, channel in enumerate(channels):
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
            self.progress(i+1, n)

        return freqs

class AverageIntermediateAction(ActionBase):
    CAPTION = "Average (static) FreqIntermediate to spectrum"
    def __call__(self, channels: list[FreqIntermediateData], average_by: AverageType=AverageType.Energy) -> list[FreqDomainData]:
        rst = []
        for channel in channels:
            rst.append(channel.to_powerspectrum(average_by=average_by))
        return rst
    
class ViewFreqDomainAction(VAB):
    CAPTION = "Show FFT spectrum"

    def __call__(self, channels: list[FreqDomainData], xlim: tuple[float, float]=None, ylim: tuple[float, float]=None, with_phase: bool=False):
        fig = self.figure
        gs = GridSpec(2, 1, height_ratios=[2, 1])

        if with_phase:
            ax = fig.add_subplot(gs[0])
            ax_p = fig.add_subplot(gs[1], sharex=ax)
            ax_p.set_ylabel("Phase [°]")
        else:
            ax = fig.gca()

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude")

        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)

        for channel in channels:
            ax.plot(channel.x, channel.amplitude, label=f"{channel.name} [{channel.y_unit}]")
            if with_phase:
                ax_p.plot(channel.x, channel.phase)

        ax.legend(loc="upper right")

class ViewFreqIntermediateAction(VAB):
    CAPTION = "Show FFT color plot"

    def __call__(self, channel: FreqIntermediateData, xlim: tuple[float, float]=None, clim: tuple[float, float]=[0, 0.001]):
        fig = self.figure
        ax = fig.gca()

        if clim is None:
            clim = [None, None]
        cmin, cmax = clim

        fig.suptitle(f"Color map: {channel.name}")
        xs = channel.x
        zs = channel.z
        ax.set_xlabel("Frequency [Hz]")
        if (ref_bins:=channel.ref_bins) is not None:
            ys = channel.ref_bins.y
            idx = np.argsort(ys)
            ys = ys[idx]
            zs = zs[idx]
            ax.set_ylabel(f"{ref_bins.name} [{ref_bins.y_unit}]")
        m = ax.pcolormesh(xs, ys, np.abs(zs), cmap='jet', vmin=cmin, vmax=cmax)
        cb = fig.colorbar(m)
        cb.set_label(f"Amplitude [{channel.z_unit}]")
        if xlim is not None:
            ax.set_xlim(xlim)

class ExtractAmplitudeAction(PAB):
    CAPTION = "Extract amplitude at frequencies"

    def __call__(self, channels: list[FreqDomainData], frequencies: list[float], line_tol: int=3):
        for channel in channels:
            idxes = np.searchsorted(frequencies, channel.x)
            for idx in idxes:
                ...

class ExtractOrderSliceAction(PAB):
    CAPTION = "Extract order slice"

    def __call__(self, channels: list[FreqIntermediateData], order: float=1, average_by: "Any"=None):
        for channel in channels:
            for bin_y, f_batch in zip(channel.ref_bins.y, channel.z):
                target_x = bin_y * order
                a = f_batch.extract_amplitude_at(target_x)

            # extract to stat_data, with refs

# calc rms

class FilterSpectrumAction(PAB):
    CAPTION = "Filter spectrum"
    def __call__(self, channels: list[FreqDomainData], bands: list[tuple[float, float]], remove: bool=True) -> list[FreqDomainData]:
        rst = []
        if remove:
            for ch in channels:
                rst.append(ch.remove_spec(bands=bands))
        else:
            for ch in channels:
                rst.append(ch.keep_spec(bands=bands))
        return rst
    
class SpectrumToTimeAction(PAB):
    CAPTION = "Convert spectrum to TimeData"
    def __call__(self, channels: list[FreqDomainData]) -> list[TimeData]:
        rst = []
        for ch in channels:
            rst.append(ch.to_timedomain())
        return rst

class SpectrumAsTimeAction(PAB):
    CAPTION = "Treate frequency spectrum as TimeData"

class EnvelopeTimeAction(PAB):
    # scipy.signal.hilbert
    pass

class LoadCaseSpectrumComparison(VAB):
    def __call__(self, loadcases: list[str], channel_name: str):
        pass

class LoadCaseFreqIntermediateAverage(VAB):
    def __call__(self, loadcases: list[str], channel_name: str, ref_case: str):
        pass

# BearingEnvelopeAnalysis = FFT + FilterSpec + ...