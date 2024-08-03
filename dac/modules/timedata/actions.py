import numpy as np
import re
from scipy import signal

from dac.core.actions import ActionBase, VAB, PAB, SAB
from . import TimeData
from .data_loader import load_tdms
from ..nvh import FilterType

class LoadAction(PAB):
    CAPTION = "Load measurement data"
    def __call__(self, fpaths: list[str], ftype: str=None) -> list[TimeData]:
        n = len(fpaths)
        rst = []
        for i, fpath in enumerate(fpaths):
            if not fpath.upper().endswith("TDMS"):
                continue
            r = load_tdms(fpath=fpath)
            rst.extend(r)
            self.progress(i+1, n)
        return rst

class TruncAction(ActionBase):
    CAPTION = "Truncate TimeData"
    def __call__(self, channels: list[TimeData], duration: tuple[float, float]=[0, 0]) -> list[TimeData]:
        rst = []
        xfrom, xto = duration

        for i, channel in enumerate(channels):
            x = channel.x
            if xto==0:
                idx_to = None
            else:
                if xto<0:
                    xto = x[-1] + xto
                idx_to = np.searchsorted(x, xto)

            idx_from = np.searchsorted(x, xfrom)
            y = channel.y[idx_from:idx_to]
            rst.append(TimeData(f"{channel.name}-Trunc", y=y, dt=channel.dt, y_unit=channel.y_unit, comment=channel.comment))
        
        return rst

class FilterAction(ActionBase):
    CAPTION = "Filter TimeData"
    def __call__(self, channels: list[TimeData], freqs: tuple[float, float], order: int=3, filter_type: FilterType=FilterType.BandPass) -> list[TimeData]:
        # with current annotation mechanism, freq as single float won't be passed inside
        rst = []

        if filter_type in (FilterType.BandPass, FilterType.BandStop):
            w = np.array(freqs)
        else:
            w = freqs[0]

        for i, channel in enumerate(channels):
            Wn = w / (channel.fs / 2)
            b, a = signal.butter(order, Wn, filter_type.value)
            y = signal.filtfilt(b, a, channel.y)

            rst.append(TimeData(name=f"{channel.name}-FiltT", y=y, dt=channel.dt, y_unit=channel.y_unit, comment=channel.comment))

        return rst

class ResampleAction(ActionBase):
    CAPTION = "Resample data to"
    def __call__(self, channels: list[TimeData], dt: float=1) -> list[TimeData]:
        rst = []
        for i, channel in enumerate(channels):
            interval = int(dt // channel.dt)
            if interval > 1:
                rst.append(TimeData(name=channel.name, y=channel.y[::interval], dt=channel.dt*interval, y_unit=channel.y_unit, comment=channel.comment))
            else:
                rst.append(channel)
        return rst

class PrepDataAction(SAB, seq=[TruncAction, ResampleAction, FilterAction]): # example sequences
    ...

class EnvelopeTimeAction(PAB):
    CAPTION = "Envelop with Hilbert transform"
    # only works when there is positive and negative part
    def __call__(self, channels: list[TimeData], restore_offset: bool=False) -> list[TimeData]:
        rst = []
        for i, channel in enumerate(channels):
            channel_y = channel.y
            offset = np.mean(channel_y)
            analytic = signal.hilbert(channel_y-offset)
            env = np.abs(analytic)
            if restore_offset:
                env += offset

            rst.append(TimeData(name=f"{channel.name}-Env", y=env, dt=channel.dt, y_unit=channel.y_unit, comment=channel.comment))

        return rst

class ShowTimeDataAction(VAB):
    CAPTION = "Show measurement data"
    def __call__(self, channels: list[TimeData], plot_dt: float=None, xlim: tuple[float, float]=None, ylim: tuple[float, float]=None):
        fig = self.figure
        fig.suptitle("Time data visualization")

        ax = fig.gca()
        ax.set_xlabel("Time [s]")
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        
        for channel in channels:
            x, y = channel.x, channel.y
            if plot_dt is not None:
                interval = int(plot_dt // channel.dt)
                if interval > 1:
                    x = x[::interval]
                    y = y[::interval]
            
            ax.plot(x, y, label=f"{channel.name} [{channel.y_unit}]")
        
        ax.legend(loc="upper right")

class CounterToTachoAction(ActionBase):
    CAPTION = "Encoder counter to tacho"
    def __call__(self, channel: TimeData, ppr: int=1024, sr_delta: float=0.1) -> TimeData:
        counter = channel.y
        delta = int(channel.fs * sr_delta)
        rpm = np.zeros(len(counter))
        rpm[delta:-delta] = (counter[2*delta:]-counter[:-2*delta]) / ppr / (2*delta*channel.dt / 60)
        rpm[:delta] = rpm[delta]
        rpm[-delta:] = rpm[-delta-1]

        return TimeData(name="Tacho", y=rpm, dt=channel.dt, y_unit="rpm")
    
class ChopOffSpikesAction(PAB):
    CAPTION = "Chop off spikes"
    def __call__(self, channels: list[TimeData], max_dydt: float, limits: tuple[float, float]=None) -> list[TimeData]:
        # `max_dydt`: maximum delta y per sec

        if limits is None:
            low_lim, high_lim = -np.inf, np.inf
        else:
            low_lim, high_lim = limits

        ret_channels = []
        n = len(channels)

        for j, channel in enumerate(channels):
            signal = channel
            orig_data = signal.y
            target_data = orig_data.copy()

            max_dy = max_dydt * signal.dt

            look4raise = True
            s, e = 0, 0

            if len(target_data)>0:
                pv = orig_data[0]
            else:
                continue
            
            for i, cv in enumerate(orig_data[1:]):
                dy = cv-pv
                if look4raise:
                    if dy > max_dy:
                        s = i # start from prev idx
                        sv = pv
                        look4raise = False
                else:
                    if np.abs(dy)<max_dy and np.abs(cv-sv)<max_dy*(i+1-s) and cv<=high_lim and cv>=low_lim:
                        look4raise = True
                        target_data[(s+1):(i+1)] = np.linspace(sv, cv, num=(i-s+2), endpoint=True)[1:-1]
                pv = cv
            
            ret_channels.append(
                TimeData(
                    name=f"{channel.name}-Chop",
                    y=target_data,
                    dt=signal.dt,
                    y_unit=signal.y_unit
                )
            )

            self.progress(j+1, n)

        return ret_channels

class PulseToAzimuthAction(ActionBase):
    CAPTION = "Pulse to azimuth"
    def __call__(self, channel: TimeData, ref_level: float, ppr: int=1, higher_as_pulse: bool=True, phase_shift: float=0, ref_channel: TimeData=None) -> TimeData:
        # `ref_channel` is for example when speed ramping, azimuth is not linear equal
        if ref_channel:
            sr_ratio = ref_channel.dt / channel.dt
            ref_data = ref_channel.y

        data = channel.y
        inpulse = data>ref_level if higher_as_pulse else data<ref_level
        indexes = np.arange(len(data))[inpulse]
        idx_diff = np.diff(indexes)
        # assert len(idx_diff) == len(indexes)-1
        idx_pulse_end = indexes[:-1][idx_diff>np.mean(idx_diff)]
        ang_data = np.zeros_like(data) * np.nan
        for i, (from_idx, to_idx) in enumerate(zip(idx_pulse_end[:-1], idx_pulse_end[1:])):
            if ref_channel:
                ref_indexes = (np.arange(from_idx, to_idx) / sr_ratio).astype(int)
                aligned_subdata = ref_data[ref_indexes]
                ang_data[from_idx:to_idx] = np.cumsum(aligned_subdata) / np.sum(aligned_subdata) * 360 + 360*i
            else:
                ang_data[from_idx:to_idx] = np.arange(to_idx-from_idx)/(to_idx-from_idx)*360 + 360*i
        
        return TimeData(
            name=f"Azi-{channel.name}", dt=channel.dt, y_unit="°",
            y=(ang_data/ppr+phase_shift)%360,
        )
    
class RefPulseToAzimuthAction(PAB):
    ... # create azimuth using reference

class OpAction(ActionBase):
    CAPTION = "Operation on TimeData"
    def __call__(self, channels: list[TimeData], op_str: str="{0}", y_unit: str="-") -> TimeData:
        # assert same dt and same length
        ys = [channel.y for channel in channels]
        y = eval(re.sub(r"{(\d+)}", r"ys[\1]", op_str))
        return TimeData("Channel-Op_ed", y=y, dt=channels[0].dt, y_unit=y_unit)