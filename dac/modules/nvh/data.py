"""Data structures for NVH (Noise, Vibration, and Harshness) analysis.

This module defines classes for representing frequency domain data 
(`FreqDomainData`), intermediate frequency data (`FreqIntermediateData`),
order information (`OrderInfo`, `OrderList`), and slice data 
(`SliceData`, `OrderSliceData`).
"""
import numpy as np
from dac.core.data import DataBase
from . import BinMethod, AverageType
from ..timedata import TimeData
from collections import namedtuple

class ProcessPackage: # bundle channels and ref_channel
    ...

class DataBins(DataBase):
    def __init__(self, name: str = None, uuid: str = None, y: np.ndarray=None, y_unit: str = "-") -> None:
        """Initializes a DataBins object.

        Parameters
        ----------
        name : str, optional
            Name of the DataBins node, by default None.
        uuid : str, optional
            UUID of the DataBins node, by default None.
        y : np.ndarray, optional
            NumPy array representing the bin values, by default None
            (which results in an empty array).
        y_unit : str, optional
            Unit of the bin values, by default "-".
        """
        super().__init__(name, uuid)

        self.y = y if y is not None else np.array([])
        self.y_unit = y_unit
        self._method = BinMethod.Mean

class FreqDomainData(DataBase):
    def __init__(self, name: str = None, uuid: str = None, y: np.ndarray=None, df: float=1, y_unit: str="-") -> None:
        """Initializes a FreqDomainData object.

        Represents data in the frequency domain (e.g., a spectrum).

        Parameters
        ----------
        name : str, optional
            Name of the FreqDomainData node, by default None.
        uuid : str, optional
            UUID of the FreqDomainData node, by default None.
        y : np.ndarray, optional
            NumPy array of complex numbers representing the spectrum,
            by default None (which results in an empty array).
        df : float, optional
            Frequency resolution (delta f) in Hz, by default 1.
        y_unit : str, optional
            Unit of the spectral values, by default "-".
        """
        super().__init__(name, uuid)
    
        self.y = y if y is not None else np.array([]) # complex number
        self.y_unit = y_unit
        self.df = df

    @property
    def x(self):
        """The frequency vector for the spectrum.

        Calculated as `np.arange(self.lines) * self.df`.

        Returns
        -------
        np.ndarray
            A NumPy array representing the frequency axis.
        """
        return np.arange(self.lines) * self.df
    
    @property
    def f(self):
        """Alias for the frequency vector `x`."""
        return self.x

    @property
    def lines(self):
        """Number of frequency lines in the spectrum.

        Returns
        -------
        int
            The length of the `y` array.
        """
        return len(self.y)

    @property
    def phase(self):
        """Phase of the spectrum in degrees.

        Calculated as `np.angle(self.y, deg=True)`.

        Returns
        -------
        np.ndarray
            A NumPy array representing the phase spectrum.
        """
        return np.angle(self.y, deg=True)

    @property
    def amplitude(self):
        """Amplitude of the spectrum.

        Calculated as `np.abs(self.y)`.

        Returns
        -------
        np.ndarray
            A NumPy array representing the amplitude spectrum.
        """
        return np.abs(self.y)
    
    def remove_spec(self, bands: list[tuple[float, float]]):
        """Zeros out specified frequency bands in the spectrum.

        Parameters
        ----------
        bands : list[tuple[float, float]]
            A list of tuples, where each tuple (ffrom, fto) defines a
            frequency band to be zeroed out.

        Returns
        -------
        FreqDomainData
            A new FreqDomainData object with the specified bands removed.
        """
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
        """Keeps only specified frequency bands in the spectrum, zeroing others.

        Parameters
        ----------
        bands : list[tuple[float, float]]
            A list of tuples, where each tuple (ffrom, fto) defines a
            frequency band to be kept.

        Returns
        -------
        FreqDomainData
            A new FreqDomainData object with only the specified bands retained.
        """
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
        """Integrates the spectrum in the frequency domain.

        Performs integration by dividing by `(j * 2 * pi * f)^order`.

        Parameters
        ----------
        order : int, optional
            The order of integration, by default 1.

        Returns
        -------
        FreqDomainData
            A new FreqDomainData object representing the integrated spectrum.
            The unit is appended with '*s' for each order of integration.
        """
        a = self.x * 1j * 2 * np.pi
        b = np.zeros(self.lines, dtype="complex")
        b[1:] = a[1:]**(-order)
        y = self.y * b

        return FreqDomainData(name=f"{self.name}-IntF", y=y, df=self.df, y_unit=self.y_unit+f"*{'s'*order}")    
    
    def effective_value(self, fmin=0, fmax=0):
        """Calculates the effective (RMS) value of the spectrum.

        .. note::
            The current implementation calculates `sqrt(sum(abs(y)^2))`,
            which is related to Parseval's theorem but might need adjustment
            for a calibrated RMS value depending on windowing and scaling factors
            not present in this method. The `fmin` and `fmax` arguments are
            defined but not used.

        Parameters
        ----------
        fmin : float, optional
            Minimum frequency for the calculation (currently unused), by default 0.
        fmax : float, optional
            Maximum frequency for the calculation (currently unused), by default 0.

        Returns
        -------
        float
            The square root of the sum of the squared absolute values of the spectrum.
        """
        # index = (freq > fmin) & (freq <= fmax)
        # effvalue = sqrt(sum(abs(value(index)*new_factor/orig_factor).^2));

        return np.sqrt(np.sum(np.abs(self.y)**2))
    
    def to_timedomain(self):
        """Converts the frequency domain data back to the time domain using IFFT.

        Assumes the input `self.y` is a single-sided spectrum. It constructs a
        double-sided spectrum before performing the IFFT.

        Returns
        -------
        TimeData
            A TimeData object representing the time-domain signal.
        """
        single_spec = self.y
        double_spec = np.concatenate([single_spec, np.conjugate(single_spec[self.lines:0:-1])]) / 2
        double_spec[0] *= 2
        # I really need to consider saving all spectrum without converting between ss and ds
        y = np.real(np.fft.ifft(double_spec * len(double_spec)))

        return TimeData(name=self.name, y=y, dt=1/(self.lines*self.df*2), y_unit=self.y_unit)
    
    def as_timedomain(self):
        """Placeholder for treating spectrum as time domain data.

        (Note: The actual implementation is missing, currently a `...` statement.)
        """
        ...

    def get_amplitudes_at(self, frequencies: list[float], lines: int=3, width: float=None) -> list[tuple[float, float]]:
        """Extracts peak amplitudes from the spectrum at or near specified frequencies.

        For each given frequency, it searches for the maximum amplitude within a
        window defined by `lines` or `width`.

        Parameters
        ----------
        frequencies : list[float]
            A list of frequencies (in Hz) at which to find amplitudes.
        lines : int, optional
            The number of frequency lines to search around each target
            frequency, by default 3. Ignored if `width` is provided.
        width : float, optional
            The frequency width (in Hz) to search around each target
            frequency. If provided, `lines` is recalculated based on this,
            by default None.

        Returns
        -------
        list[tuple[float, float] or None]
            A list of tuples. Each tuple contains (actual_frequency_of_peak, peak_value).
            If no peak is found in the window for a given frequency (e.g., empty slice),
            `None` is appended for that frequency.
        """
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
        """Initializes a FreqIntermediateData object.

        Represents intermediate frequency data, typically from STFT processing,
        often a 2D array of spectra over reference bins (e.g., time or RPM).

        Parameters
        ----------
        name : str, optional
            Name of the FreqIntermediateData node, by default None.
        uuid : str, optional
            UUID of the FreqIntermediateData node, by default None.
        z : np.ndarray, optional
            NumPy array (usually 2D, e.g., batches x window_size) of complex
            numbers representing multiple spectra, by default None (results in empty array).
        df : float, optional
            Frequency resolution (delta f) in Hz, by default 1.
        z_unit : str, optional
            Unit of the spectral values, by default "-".
        ref_bins : DataBins, optional
            A DataBins object representing the reference axis (e.g., time
            or RPM bins) for the `z` data, by default None.
        """
        super().__init__(name, uuid)

        self.z = z if z is not None else np.array([]) # batches x window_size
        self.z_unit = z_unit
        self.df = df
        self.ref_bins = ref_bins

    @property
    def x(self):
        """The frequency vector for each spectrum in the data.

        Calculated as `np.arange(self.lines) * self.df`.

        Returns
        -------
        np.ndarray
            A NumPy array representing the frequency axis.
        """
        return np.arange(self.lines) * self.df
    
    @property
    def f(self):
        """Alias for the frequency vector `x`."""
        return self.x

    def _bl(self):
        """Helper method to get the number of batches and lines from `self.z.shape`.

        Returns
        -------
        tuple[int, int]
            A tuple (batches, lines).
        """
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
        """Number of frequency lines in each spectrum.

        Returns
        -------
        int
            Number of lines.
        """
        _, lines = self._bl()
        return lines
    
    @property
    def batches(self):
        """Number of batches (spectra) in the data.

        Returns
        -------
        int
            Number of batches.
        """
        batches, _ = self._bl()
        return batches
    
    def to_powerspectrum(self, average_by: AverageType=AverageType.Energy):
        """Averages the FreqIntermediateData along the batch axis to a power spectrum.

        Parameters
        ----------
        average_by : AverageType, optional
            The averaging method. Supports AverageType.Energy
            (RMS of amplitudes) and AverageType.Linear (mean of
            amplitudes), by default AverageType.Energy.

        Returns
        -------
        FreqDomainData
            A FreqDomainData object representing the averaged spectrum.
        """
        if average_by==AverageType.Energy:
            y = np.sqrt(np.mean(np.abs(self.z)**2, axis=0))
        elif average_by==AverageType.Linear:
            y = np.mean(np.abs(self.z), axis=0)
        return FreqDomainData(name=self.name, y=y, df=self.df, y_unit=self.z_unit)
    
    def rectify_to(self, x_slice: tuple, y_slice: tuple) -> "FreqIntermediateData":
        """Resamples or re-bins the FreqIntermediateData to a new grid.

        .. note::
            The actual implementation is incomplete, currently a `pass` statement
            within the loops and returns `FreqIntermediateData` class, not an instance.

        Parameters
        ----------
        x_slice : tuple
            Tuple defining the new x-axis (frequency) bins or range.
        y_slice : tuple
            Tuple defining the new y-axis (reference) bins or range.

        Returns
        -------
        FreqIntermediateData
            A new FreqIntermediateData object on the rectified grid (intended).
        """
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
        """Extracts order slices from the FreqIntermediateData.

        For each specified order, it traces the order line across the reference bins
        and extracts the peak amplitude within a tolerance window at each point.

        Parameters
        ----------
        orders : OrderList
            An OrderList defining the orders to extract.
        line_tol : int, optional
            The number of frequency lines around the theoretical order
            frequency to search for the peak amplitude, by default 3.

        Returns
        -------
        OrderSliceData
            An OrderSliceData object containing the extracted slices.
        """
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
        """Rotates the phase of `self.z` based on a reference FreqIntermediateData.

        The operation is `conj(reference.z) * self.z / abs(reference.z)`.
        This aligns the phase of `self.z` to the phase of `reference.z`.
        The commented-out lines suggest further averaging or alternative
        calculations that are not currently implemented.

        Parameters
        ----------
        reference : FreqIntermediateData
            Another FreqIntermediateData object to use as phase reference.
            It's assumed to have the same dimensions and df as self.

        Returns
        -------
        FreqIntermediateData
            A new FreqIntermediateData object with phase rotated data.
        """
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
        """Calculates the cross-spectrum between self and a reference FreqIntermediateData.

        The cross-spectrum is calculated as `mean(conj(reference.z) * self.z, axis=0)`.
        A coherence value is also calculated but not returned with the FreqDomainData.

        Parameters
        ----------
        reference : FreqIntermediateData
            Another FreqIntermediateData object. Assumed to have
            compatible dimensions and df.

        Returns
        -------
        FreqDomainData
            A FreqDomainData object representing the mean cross-spectrum.
            (Note: Coherence is calculated but not directly returned as part of this object.)
        """
        # assert shape equals, and df, and etc.
        cross = np.conj(reference.z) * self.z
        data = np.mean(cross, axis=0)
        coh = np.sqrt(
                np.abs(data) /
                (self.to_powerspectrum().y * reference.to_powerspectrum().y)
            )
        return FreqDomainData(y=data) # what about coh?
    
    def frf(self, reference: "FreqIntermediateData"):
        """Calculates Frequency Response Functions (FRF H1 and H2 estimators).

        H1 = mean(conj(reference.z) * self.z) / mean(abs(reference.z)^2)
        H2 = mean(abs(self.z)^2) / mean(conj(self.z) * reference.z)

        .. note::
            The comments in the original code `XY/X^2 ???` and `Y^2/XY ???`
            suggest some uncertainty or a need for verification of these formulae.

        Parameters
        ----------
        reference : FreqIntermediateData
            Another FreqIntermediateData object, typically the input signal.
            `self` is typically the output signal. Assumed to have
            compatible dimensions and df.

        Returns
        -------
        tuple[FreqDomainData, FreqDomainData]
            A tuple containing two FreqDomainData objects: (FRF_H1, FRF_H2).
        """
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
        """Initializes a SliceData object.

        Represents data sliced along an order or a specific path in a
        frequency-reference map.

        Parameters
        ----------
        f : np.ndarray
            NumPy array of frequency values for the slice.
        ref : np.ndarray
            NumPy array of reference values (e.g., RPM, time) for the slice.
        amplitude : np.ndarray
            NumPy array of amplitude values for the slice.
        """
        self.f = np.array(f)
        self.ref = np.array(ref)
        self.amplitude = np.array(amplitude)

    def get_aligned_f(self):
        """Returns the slice data sorted by the frequency array `f`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple (sorted_f_array, corresponding_amplitude_array).
        """
        idx = np.argsort(self.f)
        return self.f[idx], self.amplitude[idx]
    
    def get_aligned_ref(self):
        """Returns the slice data sorted by the reference array `ref`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple (sorted_ref_array, corresponding_amplitude_array).
        """
        idx = np.argsort(self.ref)
        return self.ref[idx], self.amplitude[idx]

class OrderList(DataBase):
    def __init__(self, name: str = None, uuid: str = None, orders: list[OrderInfo]=None) -> None:
        """Initializes an OrderList object.

        Parameters
        ----------
        name : str, optional
            Name of the OrderList node, by default None.
        uuid : str, optional
            UUID of the OrderList node, by default None.
        orders : list[OrderInfo], optional
            A list of OrderInfo namedtuples, by default None (results in empty list).
        """
        super().__init__(name, uuid)
        self.orders: list[OrderInfo] = orders or []

class OrderSliceData(DataBase):
    def __init__(self, name: str = None, uuid: str = None, source: FreqIntermediateData = None) -> None:
        """Initializes an OrderSliceData object.

        Holds multiple order slices, typically extracted from a single
        FreqIntermediateData object.

        Parameters
        ----------
        name : str, optional
            Name of the OrderSliceData node, by default None.
        uuid : str, optional
            UUID of the OrderSliceData node, by default None.
        source : FreqIntermediateData, optional
            The FreqIntermediateData object from which these slices
            were extracted, by default None.
        """
        super().__init__(name, uuid)

        self.slices: dict[OrderInfo, SliceData] = {}
        self.ref_source: FreqIntermediateData = source
    
    def rectify2freqdata(self): # when speed is uneven, remove high bandwidth
        """Placeholder for rectifying order slices, e.g., for uneven speed.

        .. note::
            The actual implementation is missing, currently a `pass` statement.
        """
        pass