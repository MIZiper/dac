import numpy as np
from dac.core.data import DataBase
from . import BinMethod
from ..timedata import TimeData

class ProcessPackage: # bundle channels and ref_channel
    ...

class DataBins(DataBase):
    def __init__(self, name: str = None, uuid: str = None, y: np.ndarray=None, y_unit: str = "-") -> None:
        super().__init__(name, uuid)

        self.y = y if y is not None else np.array([])
        self.y_unit = y_unit
        self._method = BinMethod.Mean

class FreqDomainData(DataBase):
    def __init__(self, name: str = None, uuid: str = None, y: np.ndarray=None, df: float=1, y_unit: str="-") -> None:
        super().__init__(name, uuid)
    
        self.y = y if y is not None else np.array([]) # complex number
        self.y_unit = y_unit
        self.df = df

    @property
    def x(self):
        return np.arange(self.lines) * self.df
    
    @property
    def f(self):
        return self.x

    @property
    def lines(self):
        return len(self.y)

    @property
    def phase(self):
        return np.angle(self.y, deg=True)

    @property
    def amplitude(self):
        return np.abs(self.y)
    
    def integral(self, order: int=1):
        a = self.x * 1j * 2 * np.pi
        b = np.zeros(self.lines, dtype="complex")
        b[1:] = a[1:]**(-order)
        y = self.y * b

        return FreqDomainData(name=self.name, y=y, df=self.df, y_unit=self.y_unit+f"*{'s'*order}")    
    
    def effective_value(self, fmin=0, fmax=0):
        # index = (freq > fmin) & (freq <= fmax)
        # effvalue = sqrt(sum(abs(value(index)*new_factor/orig_factor).^2));

        return np.sqrt(np.sum(np.abs(self.y)**2))
    
    def to_timedomain(self):
        single_spec = self.y
        double_spec = np.concatenate(single_spec, np.conjugate(single_spec[self.lines:0:-1]))
        y = np.real(np.fft.ifft(double_spec))

        return TimeData(name=self.name, y=y, dt=1/(self.lines*self.df*2), y_unit=self.y_unit)

class FreqIntermediateData(DataBase):
    def __init__(self, name: str = None, uuid: str = None, z: np.ndarray=None, df: float=1, z_unit: str="-", ref_bins: DataBins=None) -> None:
        super().__init__(name, uuid)

        self.z = z if z is not None else np.array([]) # batches x window_size
        self.z_unit = z_unit
        self.df = df
        self.ref_bins = ref_bins

    @property
    def x(self):
        return np.arange(self.lines) * self.df
    
    @property
    def f(self):
        return self.x

    def _bl(self):
        if len(shape:=self.z.shape)==0:
            # shape == ()
            batches, lines = 0, 0
        elif len(shape) == 1: # np.array([p1, p2, p3, ...])
            batches, lines = 1, shape[0]
        else:
            batches, lines = shape

        return batches, lines

    @property
    def lines(self):
        _, lines = self._bl()
        return lines
    
    @property
    def batches(self):
        batches, _ = self._bl()
        return batches

class OrderSliceData:
    ...