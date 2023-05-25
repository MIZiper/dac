from dac.core.data import DataBase
import numpy as np

class TimeData(DataBase):
    def __init__(self, name: str, y: np.array, dt: float=1, y_unit: str="-", comment: str="") -> None:
        super().__init__(name)

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