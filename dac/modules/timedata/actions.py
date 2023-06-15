import numpy as np
from scipy import signal

from dac.core.actions import ActionBase, VAB, PAB, SAB
from . import TimeData
from .data_loader import load_tdms
from ..nvh import FilterType

class LoadAction(PAB):
    CAPTION = "Load measurement data"
    def __call__(self, fpaths: list[str], ftype: str=None) -> list[TimeData]: # fpath->fpaths?
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
    def __call__(self, channels: list[TimeData], duration: tuple[float, float]=(0, 0)):
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
            rst.append(TimeData(channel.name, y=y, dt=channel.dt, y_unit=channel.y_unit, comment=channel.comment))
        
        return rst

class FilterAction(ActionBase):
    def __call__(self, channels: list[TimeData], freq: tuple[float, float], order: int=3, filter_type: FilterType=FilterType.LowPass):
        rst = []

        if filter_type in (FilterType.BandPass, FilterType.BandStop):
            w = np.array(freq)
        else:
            w = freq

        for i, channel in enumerate(channels):
            Wn = w / (channel.fs / 2)
            b, a = signal.butter(order, Wn, filter_type.value)
            y = signal.filtfilt(b, a, channel.y)

            rst.append(TimeData(name=channel.name, y=y, dt=channel.dt, y_unit=channel.y_unit, comment=channel.comment))

        return rst

class ResampleAction(ActionBase):
    def __call__(self, channels: list[TimeData], dt: float=1):
        rst = []
        for i, channel in enumerate(channels):
            interval = dt // channel.dt 
            if interval > 1:
                rst.append(TimeData(name=channel.name, y=channel.y[::interval], dt=channel.dt*interval, y_unit=channel.y_unit, comment=channel.comment))
            else:
                rst.append(channel)
        return rst

class PrepDataAction(SAB, seq=[TruncAction, ResampleAction, FilterAction]): # example sequences
    ...

# counter to rotational speed

class ShowTimeDataAction(VAB):
    CAPTION = "Show measurement data"
    def __call__(self, channels: list[TimeData], plot_dt: float=None):
        ax = self.figure.gca()
        
        for channel in channels:
            if plot_dt is not None:
                ...
            else:
                ax.plot(channel.x, channel.y, label=f"{channel.name} [{channel.y_unit}]")
        
        ax.legend(loc="upper right")