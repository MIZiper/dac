import numpy as np

from dac.core.actions import ActionBase, VAB
from . import TimeData

class LoadDataAction(ActionBase):
    CAPTION = "Load measurement data"
    def __call__(self, fpath: str, ftype: str=None) -> list[TimeData]:
        ...

class TruncDataAction(ActionBase):
    CAPTION = "Truncate TimeData"
    def __call__(self, channels: list[TimeData], duration: tuple[float, float]=(0, 0)):
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