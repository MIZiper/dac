import numpy as np
from dac.core.data import DataBase
from . import BinMethod, AverageType
from ..timedata import TimeData
from collections import namedtuple

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
    
    def remove_spec(self, bands: list[tuple[float, float]]):
        y = self.y.copy()
        x = self.x

        for ffrom, fto in bands:
            b = np.all([ffrom<=x, x<=fto], axis=0)
            y[b] = 0

        return FreqDomainData(
            name=f"{self.name}-FiltF",
            y=y,
            df=self.df,
            y_unit=self.y_unit
        )
    
    def keep_spec(self, bands: list[tuple[float, float]]):
        y = np.zeros_like(self.y)
        x = self.x

        for ffrom, fto in bands:
            b = np.all([ffrom<=x, x<=fto], axis=0)
            y[b] = self.y[b]

        return FreqDomainData(
            name=f"{self.name}-ExtractF",
            y=y,
            df=self.df,
            y_unit=self.y_unit
        )
    
    def integral(self, order: int=1):
        a = self.x * 1j * 2 * np.pi
        b = np.zeros(self.lines, dtype="complex")
        b[1:] = a[1:]**(-order)
        y = self.y * b

        return FreqDomainData(name=f"{self.name}-IntF", y=y, df=self.df, y_unit=self.y_unit+f"*{'s'*order}")    
    
    def effective_value(self, fmin=0, fmax=0):
        # index = (freq > fmin) & (freq <= fmax)
        # effvalue = sqrt(sum(abs(value(index)*new_factor/orig_factor).^2));

        return np.sqrt(np.sum(np.abs(self.y)**2))
    
    def to_timedomain(self):
        single_spec = self.y
        double_spec = np.concatenate([single_spec, np.conjugate(single_spec[self.lines:0:-1])]) / 2
        double_spec[0] *= 2
        # I really need to consider saving all spectrum without converting between ss and ds
        y = np.real(np.fft.ifft(double_spec * len(double_spec)))

        return TimeData(name=self.name, y=y, dt=1/(self.lines*self.df*2), y_unit=self.y_unit)
    
    def as_timedomain(self):
        ...

    def get_amplitudes_at(self, frequencies: list[float], lines: int=3, width: float=None) -> list[tuple[float, float]]:
        if width is not None:
            lines = int(np.ceil(width / self.df))

        x = self.x
        y = self.y
        fas = []

        for f in frequencies:
            i = np.searchsorted(x, f)
            y_p = y[max((i-lines), 0):(i+lines)] # i-lines can <0, and (i-lines):(i+lines) return empty
            x_p = x[max((i-lines), 0):(i+lines)]
            if len(y_p)==0:
                fas.append(None)
                continue
            i_p = np.argmax(np.abs(y_p))
            fas.append( (x_p[i_p], y_p[i_p],) )

        return fas


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
    
    def to_powerspectrum(self, average_by: AverageType=AverageType.Energy):
        if average_by==AverageType.Energy:
            y = np.sqrt(np.mean(np.abs(self.z)**2, axis=0))
        elif average_by==AverageType.Linear:
            y = np.mean(np.abs(self.z), axis=0)
        return FreqDomainData(name=self.name, y=y, df=self.df, y_unit=self.z_unit)
    
    def rectify_to(self, x_slice: tuple, y_slice: tuple) -> "FreqIntermediateData":
        ref_bins = self.ref_bins
        ys = ref_bins.y
        idx = np.argsort(ys)
        ys = ys[idx]
        zs = self.z[idx]

        xs = self.x # the frequencies

        x_bins = np.arange(x_slice)
        x_idxes = np.digitize(xs, x_bins)
        y_bins = np.arange(y_bins)
        y_idxes = np.digitize(ys, y_bins)

        # average by energy
        for y in ys:
            for x in xs:
                pass

        return FreqIntermediateData
    
    def extract_orderslice(self, orders: "OrderList", line_tol: int=3) -> "OrderSliceData":
        xs = self.x # [Hz]
        ys = self.ref_bins.y
        zs = self.z # batch x window

        idx = np.argsort(ys)
        ys = ys[idx]
        zs = zs[idx]

        order_slice = OrderSliceData(name="OrderSlice", source=self)

        for order in orders.orders:
            # slice_element
            se_x = [] # frequency, [Hz]
            se_y = [] # ref value, e.g. [rpm]
            se_z = [] # amplitude, e.g. [mm/s]

            for ref_y, f_batch in zip(ys, zs):
                target_x = ref_y * order.value
                target_idx = np.searchsorted(xs, target_x)

                # TODO: avoid f(0)
                # TODO: avoid out-of-range f
                rel_idx = np.argmax(np.abs(f_batch[max(target_idx-line_tol,0):(target_idx+line_tol)]))
                final_a = f_batch[target_idx-line_tol+rel_idx]
                final_f = xs[target_idx-line_tol+rel_idx]

                se_x.append(final_f)
                # if f already in orderslice? if f not ascending?
                se_y.append(ref_y)
                se_z.append(np.abs(final_a))
                # how to average? by energy

            order_slice.slices[order] = SliceData(f=se_x, ref=se_y, amplitude=se_z)

        return order_slice

    def reference_to(self, reference: "FreqIntermediateData"):
        data = np.conj(reference.z) * self.z / np.abs(reference.z)
        # it's actually rotate self with reference angle
        
        # data = np.mean(data, axis=0)
        # # no linear here, 'cause we can do that later
        # # new object . to_powerspectrum(AverageType.Linear)

        # # calc_phrefspectrum2
        # # I don't know the meaning / scene
        # # it's kind of a different average type
        # data = (
        #         np.mean(np.conj(reference.data) * self.data, axis=0) /
        #         np.sqrt(np.mean(np.abs(reference.data)**2, axis=0))
        #     )

        return FreqIntermediateData(z=data)
    
    def cross_spectrum_with(self, reference: "FreqIntermediateData"):
        
        # assert shape equals, and df, and etc.
        cross = np.conj(reference.z) * self.z
        data = np.mean(cross, axis=0)
        coh = np.sqrt(
                np.abs(data) /
                (self.to_powerspectrum().y * reference.to_powerspectrum().y)
            )
        return FreqDomainData(y=data) # what about coh?
    
    def frf(self, reference: "FreqIntermediateData"):
        cross12 = np.conj(reference.z) * self.z
        cross21 = np.conj(self.z) * reference.z

        spectr1 = np.abs(self.z)**2
        spectr2 = np.abs(reference.z)**2

        frfH1 = np.mean(cross12, axis=0) / np.mean(spectr2, axis=0) # XY/X^2 ???
        frfH2 = np.mean(spectr1, axis=0) / np.mean(cross21, axis=0) # Y^2/XY ???
        # need some theory about: XY v.s. YX
        # and why spectr2 as X^2, spectr1 as Y^2

        return (
            FreqDomainData(y=frfH1),
            FreqDomainData(y=frfH2)
        )

OrderInfo = namedtuple("OrderInfo", ['name', 'value', 'disp_value'])
    # name: label, e.g. f_1
    # value: ratio between reference and frequency
    # disp_value: for the case of unit conversion, e.g. 1st order of 1 [rpm] is actually 1/60 [Hz]

class SliceData:
    def __init__(self, f: np.ndarray, ref: np.ndarray, amplitude: np.ndarray):
        self.f = np.array(f)
        self.ref = np.array(ref)
        self.amplitude = np.array(amplitude)

    def get_aligned_f(self):
        idx = np.argsort(self.f)
        return self.f[idx], self.amplitude[idx]
    
    def get_aligned_ref(self):
        idx = np.argsort(self.ref)
        return self.ref[idx], self.amplitude[idx]

class OrderList(DataBase):
    def __init__(self, name: str = None, uuid: str = None, orders: list[OrderInfo]=None) -> None:
        super().__init__(name, uuid)
        self.orders: list[OrderInfo] = orders or []

class OrderSliceData(DataBase):
    def __init__(self, name: str = None, uuid: str = None, source: FreqIntermediateData = None) -> None:
        super().__init__(name, uuid)

        self.slices: dict[OrderInfo, SliceData] = {}
        self.ref_source: FreqIntermediateData = source
    
    def rectify2freqdata(self): # when speed is uneven, remove high bandwidth
        pass