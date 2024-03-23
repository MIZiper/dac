from dac.core.data import DataBase
import numpy as np

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
        # # corresponding values are connected

        return batches
    
    def effective_value(self):
        return np.sqrt(np.mean(self.y**2))