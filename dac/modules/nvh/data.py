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
        """Represents the frequency spectrum's amplitudes as a TimeData object.
        The time axis is index-based (dt=1.0). For a full inverse FFT
        conversion to the time domain, use the `to_timedomain()` method.
        """
        amplitudes = np.abs(self.y)
        return TimeData(
            name=f"{self.name}_as_time",
            y=amplitudes,
            dt=1.0,
            y_unit=self.y_unit
        )

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

        # x_bins = np.arange(x_slice) # This was a placeholder
        # x_idxes = np.digitize(xs, x_bins)
        # y_bins = np.arange(y_bins) # This was a placeholder
        # y_idxes = np.digitize(ys, y_bins)

        # average by energy
        # for y_val in ys: # Renamed to y_val to avoid conflict with outer scope ys
        #     for x_val in xs: # Renamed to x_val to avoid conflict with outer scope xs
        #         pass

        # Implementation based on the detailed plan:

        # 1. Handle empty or invalid input
        if self.z is None or self.z.size == 0:
            return FreqIntermediateData(name=f"{self.name}_rectified_emptyZ", z_unit=self.z_unit, df=self.df, ref_bins=self.ref_bins)
        if self.ref_bins is None or self.ref_bins.y is None or self.ref_bins.y.size == 0:
            return FreqIntermediateData(name=f"{self.name}_rectified_emptyRef", z_unit=self.z_unit, df=self.df)

        if not (isinstance(x_slice, tuple) and len(x_slice) == 3 and x_slice[2] > 0):
            print("Warning: Invalid x_slice. Must be (start, stop, num_points > 0).")
            return FreqIntermediateData(name=f"{self.name}_rectified_invalidXSlice", z_unit=self.z_unit, df=self.df, ref_bins=self.ref_bins)
        if not (isinstance(y_slice, tuple) and len(y_slice) == 3 and y_slice[2] > 0):
            print("Warning: Invalid y_slice. Must be (start, stop, num_points > 0).")
            return FreqIntermediateData(name=f"{self.name}_rectified_invalidYSlice", z_unit=self.z_unit, df=self.df, ref_bins=self.ref_bins)
        
        if x_slice[2] == 1 and x_slice[0] != x_slice[1]: # Special case for linspace if num_points is 1
             print("Warning: x_slice num_points is 1, this might lead to df=0 or unexpected binning. Ensure start=stop for single point.")
        if y_slice[2] == 1 and y_slice[0] != y_slice[1]:
             print("Warning: y_slice num_points is 1, this might lead to issues. Ensure start=stop for single point.")


        # 2. Define the new grid bin edges and centers
        # For num_points cells, we need num_points + 1 edges.
        new_x_bins = np.linspace(x_slice[0], x_slice[1], int(x_slice[2]) + 1)
        new_y_bins = np.linspace(y_slice[0], y_slice[1], int(y_slice[2]) + 1)
        
        new_x_centers = (new_x_bins[:-1] + new_x_bins[1:]) / 2 if x_slice[2] > 0 else np.array([])
        new_y_centers = (new_y_bins[:-1] + new_y_bins[1:]) / 2 if y_slice[2] > 0 else np.array([])

        # 3. Prepare original data
        orig_x_values = self.x  # frequencies
        orig_y_values = self.ref_bins.y
        orig_z_values = self.z

        if not np.all(np.diff(orig_y_values) >= 0): # Check if y is sorted
            sort_idx = np.argsort(orig_y_values)
            orig_y_values = orig_y_values[sort_idx]
            orig_z_values = orig_z_values[sort_idx, :]

        # 4. Initialize the new Z data array
        # The dimensions of new_z_data should be (num_y_points, num_x_points)
        num_new_y_points = int(y_slice[2])
        num_new_x_points = int(x_slice[2])
        new_z_data_sq_sum = np.zeros((num_new_y_points, num_new_x_points))
        counts = np.zeros_like(new_z_data_sq_sum, dtype=int)

        # 5. Assign original data points to new bins and accumulate
        for i, y_val in enumerate(orig_y_values):
            # np.digitize returns 1-based index. Bins are [edge1, edge2), [edge2, edge3)...
            # Values < new_y_bins[0] get 0, values >= new_y_bins[-1] get len(new_y_bins)
            # We want 0-based index for our new_z_data array.
            y_bin_idx = np.digitize(y_val, new_y_bins[1:-1]) # Digitize against inner edges for 0 to N-1 mapping
                                                            # Or, more simply, use all edges and adjust index
            
            # Correct indexing for np.digitize:
            # It returns the index of the bin (1-based) each element belongs to.
            # For N bins (N+1 edges), it returns values from 1 to N.
            # If value is < edges[0], it returns 0.
            # If value is >= edges[-1], it returns N+1 (or len(edges)).
            # We map these to 0 to num_new_y_points-1
            
            temp_y_bin_idx = np.digitize(y_val, new_y_bins)
            if new_y_bins[0] <= y_val < new_y_bins[-1]: # Check if y_val is within the span of new_y_bins (exclusive of last edge)
                y_bin_idx = temp_y_bin_idx - 1 # Convert 1-based to 0-based
                if not (0 <= y_bin_idx < num_new_y_points): # Should be caught by the outer check mostly
                    continue
            elif y_val == new_y_bins[-1] and num_new_y_points > 0: # Value exactly at the last edge, put in last bin
                 y_bin_idx = num_new_y_points - 1
            else: # Outside the range of new_y_bins
                continue


            current_z_row = orig_z_values[i, :]
            for j, x_val in enumerate(orig_x_values):
                temp_x_bin_idx = np.digitize(x_val, new_x_bins)
                if new_x_bins[0] <= x_val < new_x_bins[-1]:
                    x_bin_idx = temp_x_bin_idx - 1
                    if not (0 <= x_bin_idx < num_new_x_points):
                        continue
                elif x_val == new_x_bins[-1] and num_new_x_points > 0: # Value exactly at the last edge
                    x_bin_idx = num_new_x_points - 1
                else: # Outside the range of new_x_bins
                    continue
                
                new_z_data_sq_sum[y_bin_idx, x_bin_idx] += np.abs(current_z_row[j])**2
                counts[y_bin_idx, x_bin_idx] += 1
        
        # 6. Finalize average
        averaged_z_data = np.zeros_like(new_z_data_sq_sum)
        valid_counts_mask = counts > 0
        averaged_z_data[valid_counts_mask] = np.sqrt(new_z_data_sq_sum[valid_counts_mask] / counts[valid_counts_mask])
        # For counts == 0, averaged_z_data remains 0, which is fine. Or use np.nan if preferred.
        # averaged_z_data[~valid_counts_mask] = np.nan # Optional: use NaN for empty bins

        # 7. Create and return the new FreqIntermediateData object
        new_ref_bins_y = new_y_centers
        new_ref_bins_name = self.ref_bins.name if self.ref_bins and self.ref_bins.name else "ref_rectified"
        new_ref_bins_unit = self.ref_bins.y_unit if self.ref_bins else ""
        
        new_ref_bins = DataBins(name=f"{new_ref_bins_name}_rectified", y=new_ref_bins_y, y_unit=new_ref_bins_unit)
        
        if len(new_x_centers) > 1:
            new_df = new_x_centers[1] - new_x_centers[0]
        elif len(new_x_centers) == 1: # Single X point
            new_df = 0 # Or some other appropriate value, like original df if x_slice range was 0
        else: # No X points
            new_df = self.df # Fallback or could be 0

        return FreqIntermediateData(
            name=f"{self.name}_rectified",
            z=averaged_z_data,
            df=new_df,
            z_unit=self.z_unit,
            ref_bins=new_ref_bins
        )

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
    
    def rectify2freqdata(self, order_name: str, num_freq_points: int = 512) -> "FreqDomainData":
        """
        Rectifies the amplitude data of a specific order slice onto a new, regular frequency axis.
        This means it resamples the (frequency, amplitude) pairs from the SliceData
        onto a linearly spaced frequency axis.

        Args:
            order_name (str): The name of the order to rectify (matches OrderInfo.name).
            num_freq_points (int): The number of points for the new regular frequency axis.

        Returns:
            FreqDomainData: A new FreqDomainData object with the rectified spectrum.
                            Returns an empty or placeholder FreqDomainData if the order
                            is not found, data is empty, or resampling is not possible.
        """
        target_slice_data: SliceData = None
        target_order_info: OrderInfo = None

        for order_info, slice_data_item in self.slices.items():
            if order_info.name == order_name:
                target_slice_data = slice_data_item
                target_order_info = order_info
                break
        
        y_unit = self.ref_source.z_unit if self.ref_source and hasattr(self.ref_source, 'z_unit') else "-"
        output_base_name = f"{self.name}_{order_name}_rectified"

        if target_slice_data is None:
            print(f"Warning: Order name '{order_name}' not found in OrderSliceData '{self.name}'.")
            return FreqDomainData(name=f"{output_base_name}_not_found", y_unit=y_unit)

        slice_frequencies = target_slice_data.f
        slice_amplitudes = target_slice_data.amplitude

        if slice_frequencies is None or slice_frequencies.size == 0 or \
           slice_amplitudes is None or slice_amplitudes.size == 0:
            print(f"Warning: SliceData for order '{order_name}' is empty.")
            return FreqDomainData(name=f"{output_base_name}_empty_slice", y_unit=y_unit)

        f_min = np.min(slice_frequencies)
        f_max = np.max(slice_frequencies)
        df = 0
        new_amplitudes = np.zeros(num_freq_points, dtype=float)

        if f_min == f_max:
            if num_freq_points == 1:
                new_amplitudes[0] = np.mean(slice_amplitudes)
                # df remains 0, new_freq_axis would be [f_min]
            else:
                print(f"Warning: Cannot rectify order '{order_name}' with a single frequency point ({f_min} Hz) to {num_freq_points} points.")
                return FreqDomainData(name=f"{output_base_name}_single_freq_error", y_unit=y_unit, df=0)
        else: # f_min != f_max
            if num_freq_points <= 1: # Should be at least 2 points to define a range for linspace/df
                 print(f"Warning: num_freq_points ({num_freq_points}) must be > 1 for a frequency range. Defaulting to 2.")
                 num_freq_points = 2 # Ensure at least 2 points if there's a range.
            
            # Define bin edges for mapping original frequencies to new bins
            # We'll have num_freq_points bins, so num_freq_points+1 edges.
            # The new_freq_axis will represent the centers of these bins.
            bin_edges = np.linspace(f_min, f_max, num_freq_points + 1)
            df = (f_max - f_min) / (num_freq_points -1) if num_freq_points > 1 else 0


            new_amplitudes_sum = np.zeros(num_freq_points, dtype=float)
            counts = np.zeros(num_freq_points, dtype=int)
            
            # Assign original frequencies to the new bins
            # np.digitize returns 1-based indices. bin_edges[j-1] <= x < bin_edges[j] -> j
            assigned_bin_indices = np.digitize(slice_frequencies, bin_edges)

            for i in range(len(slice_frequencies)):
                original_freq = slice_frequencies[i]
                original_amp = slice_amplitudes[i]
                
                # Correct bin_idx: 0 to num_freq_points-1
                # If freq == f_min, digitize might give 1.
                # If freq == f_max, digitize might give num_freq_points+1.
                bin_idx = assigned_bin_indices[i] - 1

                if original_freq == f_max: # Value exactly at the last edge, put in last bin
                    bin_idx = num_freq_points - 1
                
                if 0 <= bin_idx < num_freq_points:
                    new_amplitudes_sum[bin_idx] += original_amp
                    counts[bin_idx] += 1
            
            mask = counts > 0
            new_amplitudes[mask] = new_amplitudes_sum[mask] / counts[mask]
            # For bins with no points, new_amplitudes remains 0.0

        # FreqDomainData expects complex numbers for y
        y_complex = np.array(new_amplitudes, dtype=complex)

        return FreqDomainData(
            name=output_base_name,
            y=y_complex,
            df=df,
            y_unit=y_unit
        )