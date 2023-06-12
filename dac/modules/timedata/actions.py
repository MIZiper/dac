import numpy as np

from dac.core.actions import ActionBase, VAB, PAB, SAB
from . import TimeData
from .data_loader import load_tdms

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
        ...

class FilterAction(ActionBase):
    ...

class ResampleAction(ActionBase):
    ...

class PrepDataAction(SAB, seq=[TruncAction, ResampleAction, FilterAction]): # example sequences
    ...

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