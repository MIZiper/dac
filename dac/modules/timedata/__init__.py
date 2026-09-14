"""Defines the `TimeData` class mainly for high sample rate measurement data.
"""

from dac.core.data import DataBase
import numpy as np
from scipy.integrate import cumulative_trapezoid

class TimeData(DataBase):
    def __init__(self, name: str = None, uuid: str = None, y: np.ndarray=None, dt: float=1, y_unit: str="-", comment: str="") -> None:
        super().__init__(name, uuid)

        self.y = y if y is not None else np.array([])
        self.dt = dt
        self.y_unit = y_unit
        self.comment = comment
        # t0

    @property
    def fs(self):
        return 1/self.dt
    
    @property
    def length(self):
        return len(self.y)
    
    @property
    def x(self):
        return np.arange(self.length) * self.dt
    
    @property
    def t(self):
        return self.x # combine with t0
    
    def to_bins(self, df: float, overlap: float) -> np.ndarray:
        y = self.y
        batch_N = int( 1/df * self.fs )
        stride_N = int( batch_N * (1-overlap) )
        N_batches = (self.length-batch_N) // stride_N + 1
        stride, = y.strides
        assert N_batches > 0

        batches = np.lib.stride_tricks.as_strided(y, shape=(N_batches, batch_N), strides=(stride*stride_N, stride))
        # batches -= batches_mean
        # # the code above will cause problem, it's a `as_strided` mapping
        # # # corresponding values are connected

        return batches
    
    def effective_value(self):
        return np.sqrt(np.mean(self.y**2))

    def _index_range(self, t_start: float = None, t_end: float = None) -> tuple[int, int]:
        """Returns the clamped sample index range ``[i_start, i_end)`` for a time span.

        ``None`` means open-ended (from the beginning / to the end).
        """
        x = self.x
        i_start = np.searchsorted(x, t_start, side="left") if t_start is not None else 0
        i_end = np.searchsorted(x, t_end, side="right") if t_end is not None else self.length
        i_start = int(np.clip(i_start, 0, self.length))
        i_end = int(np.clip(i_end, 0, self.length))
        return i_start, i_end

    def select_range(self, t_start: float = None, t_end: float = None) -> "TimeData":
        """Returns a new `TimeData` containing only the ``[t_start, t_end]`` span.

        Times are in seconds (see `x`). ``None`` leaves that end open.
        """
        i_start, i_end = self._index_range(t_start, t_end)
        return TimeData(
            name=f"{self.name}-Sel",
            y=self.y[i_start:i_end],
            dt=self.dt,
            y_unit=self.y_unit,
            comment=self.comment,
        )

    def integrate(self):
        y_int = cumulative_trapezoid(self.y, self.x, initial=0)
        return TimeData(
            name=f"{self.name}-IntT",
            y=y_int,
            dt=self.dt,
            y_unit=f"{self.y_unit}*s",
            comment=self.comment,
        )

    def differentiate(self):
        y_diff = np.gradient(self.y, self.dt)
        return TimeData(
            name=f"{self.name}-DiffT",
            y=y_diff,
            dt=self.dt,
            y_unit=f"{self.y_unit}/s",
            comment=self.comment,
        )

    def statistics(self, t_range: tuple[float, float] = None) -> dict:
        """Computes statistics, optionally restricted to ``t_range=(start, end)`` seconds."""
        y = self.y
        if t_range is not None:
            i_start, i_end = self._index_range(*t_range)
            y = y[i_start:i_end]

        if len(y) == 0:
            return {
                "name": self.name,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "rms": float("nan"),
                "crest_factor": float("nan"),
                "skewness": float("nan"),
                "kurtosis": float("nan"),
            }

        rms = np.sqrt(np.mean(y**2))
        return {
            "name": self.name,
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "rms": float(rms),
            "crest_factor": float(np.max(np.abs(y)) / rms) if rms > 0 else 0.0,
            "skewness": float((np.mean((y - np.mean(y))**3)) / (np.std(y)**3 + 1e-30)),
            "kurtosis": float((np.mean((y - np.mean(y))**4)) / (np.std(y)**4 + 1e-30)),
        }