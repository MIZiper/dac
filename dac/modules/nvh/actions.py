import numpy as np
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import TextBox, RadioButtons
from matplotlib.axes import Axes

from dac.core.data import SimpleDefinition
from dac.core.actions import ActionBase, VAB, PAB, SAB
from dac.modules.timedata import TimeData
from . import WindowType, BandCorrection, BinMethod, AverageType, ToleranceType
from .data import FreqIntermediateData, DataBins, FreqDomainData, \
                  OrderInfo, OrderList, SliceData, OrderSliceData

class ToFreqDomainAction(PAB):
    CAPTION = "Simple FFT to frequency domain" # rect window

    def __call__(self, channels: list[TimeData], window: WindowType=WindowType.Uniform, corr: BandCorrection=BandCorrection.NarrowBand) -> list[FreqDomainData]:
        rst = []
        m = len(channels)
        window_funcs = {
            WindowType.Hanning: np.hanning,
            WindowType.Hamming: np.hamming,
        }
        for i, ch in enumerate(channels):
            batch_N = ch.length
            df = 1 / (batch_N*ch.dt)
            
            if window in window_funcs:
                windowed_y = ch.y * window_funcs[window](batch_N)
            else:
                window = WindowType.Uniform
                windowed_y = ch.y
            fdata = np.fft.fft(windowed_y) / batch_N * window.value[corr.value]

            double_spec = fdata[:int(np.ceil(batch_N/2))]
            double_spec[1:] *= 2

            rst.append(FreqDomainData(name=ch.name, y=double_spec, df=df, y_unit=ch.y_unit))
            self.progress(i+1, m)
        return rst

class ToFreqIntermediateAction(PAB):
    CAPTION = "FFT to frequency domain with window and reference"

    def __call__(self, channels: list[TimeData],
                 window: WindowType=WindowType.Hanning, corr: BandCorrection=BandCorrection.NarrowBand,
                 resolution: float=0.5, overlap: float=0.75,
                 ref_channel: TimeData=None,
                ) -> list[FreqIntermediateData]:
        
        freqs = []

        window_funcs = {
            WindowType.Hanning: np.hanning,
            WindowType.Hamming: np.hamming,
        }

        if ref_channel is not None:
            ref_batches = ref_channel.to_bins(df=resolution, overlap=overlap)
            ref_bins_y = np.mean(ref_batches, axis=1)
            ref_bins = DataBins(name=ref_channel.name, y=ref_bins_y, y_unit=ref_channel.y_unit)
        # else:
        #     create a TimeData channel, but don't know the length

        n = len(channels)
        for i, channel in enumerate(channels):
            batches = channel.to_bins(df=resolution, overlap=overlap)
            N_batches, batch_N = batches.shape

            if ref_channel is None:
                ref_bins_y = np.arange(N_batches) * 1/resolution * (1-overlap)
                ref_bins = DataBins(name="Time", y=ref_bins_y, y_unit="s")
                ref_bins._method = BinMethod.Min

            batches = batches * window_funcs[window](batch_N)
            batches_fft = np.fft.fft(batches) / batch_N * window.value[corr.value]

            double_spec = batches_fft[:, :int(np.ceil(batch_N/2))]
            double_spec[:, 1:] *= 2

            freq = FreqIntermediateData(name=channel.name, z=double_spec, df=resolution, z_unit=channel.y_unit, ref_bins=ref_bins)
            freqs.append(freq)
            self.progress(i+1, n)

        return freqs

class AverageIntermediateAction(ActionBase):
    CAPTION = "Average (static) FreqIntermediate to spectrum"
    def __call__(self, channels: list[FreqIntermediateData], average_by: AverageType=AverageType.Energy) -> list[FreqDomainData]:
        rst = []
        for channel in channels:
            rst.append(channel.to_powerspectrum(average_by=average_by))
        return rst
    
class ViewFreqDomainAction(VAB):
    CAPTION = "Show FFT spectrum"

    def __call__(self, channels: list[FreqDomainData], xlim: tuple[float, float]=None, ylim: tuple[float, float]=None, with_phase: bool=False):
        fig = self.figure
        gs = GridSpec(2, 1, height_ratios=[2, 1])

        if with_phase:
            ax = fig.add_subplot(gs[0])
            ax_p = fig.add_subplot(gs[1], sharex=ax)
            ax_p.set_ylabel("Phase [°]")
        else:
            ax = fig.gca()

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude")

        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)

        for channel in channels:
            ax.plot(channel.x, channel.amplitude, label=f"{channel.name} [{channel.y_unit}]")
            if with_phase:
                ax_p.plot(channel.x, channel.phase)

        ax.legend(loc="upper right")

class ViewFreqIntermediateAction(VAB):
    CAPTION = "Show FFT color plot"

    def __call__(self, channel: FreqIntermediateData, xlim: tuple[float, float]=None, clim: tuple[float, float]=[0, 0.001]):
        fig = self.figure
        ax = fig.gca()

        if clim is None:
            clim = [None, None]
        cmin, cmax = clim

        fig.suptitle(f"Color map: {channel.name}")
        xs = channel.x
        zs = channel.z
        ax.set_xlabel("Frequency [Hz]")
        if (ref_bins:=channel.ref_bins) is not None:
            ys = channel.ref_bins.y
            idx = np.argsort(ys)
            ys = ys[idx]
            zs = zs[idx]
            ax.set_ylabel(f"{ref_bins.name} [{ref_bins.y_unit}]")
        m = ax.pcolormesh(xs, ys, np.abs(zs), cmap='jet', vmin=cmin, vmax=cmax)
        cb = fig.colorbar(m)
        cb.set_label(f"Amplitude [{channel.z_unit}]")
        if xlim is not None:
            ax.set_xlim(xlim)

class ExtractAmplitudeAction(PAB):
    CAPTION = "Extract amplitude at frequencies"

    def __call__(self, channels: list[FreqDomainData], frequencies: list[float], line_tol: int=3):
        ...

class ViewColorPlotAndCheckOrderSlice(ViewFreqIntermediateAction):
    # obsolete, it's too laggy
    CAPTION = "Show color plot with order indication"
    def __call__(self, channel: FreqIntermediateData, xlim: tuple[float, float] = None, clim: tuple[float, float] = [0, 0.001]):
        super().__call__(channel, xlim, clim)
        fig = self.figure
        ax: Axes = fig.axes[0]
        
        ax_order = fig.add_axes([0.1, 0.01, 0.5, 0.05])
        order_input = TextBox(ax_order, label="Order:", initial=0)

        l = ax.axline((0, 0), slope=0, color='k')

        def on_press(event):
            if fig.canvas.widgetlock.locked():
                return            
            order_input.set_val(f"{event.ydata/event.xdata:.3f}")

        def on_order_submit(order_str: str):
            order = float(order_str)
            if order<=0:
                return
            l.set_slope(order)
            fig.canvas.draw_idle()
            
        order_input.on_submit(on_order_submit)

        self._cids.append(
            fig.canvas.mpl_connect('button_press_event', on_press)
        )
        self._widgets.append(order_input)

class MarkOrders(VAB):
    pass

class ViewColorPlotWithOrderSlice(ViewFreqIntermediateAction): # inherit SAB = ViewColorPlot + MarkOrders
    CAPTION = "Show color plot with order indication"
    def __call__(self, channel: FreqIntermediateData, orders: OrderList, fmt_lines: list[str]=["{f_1}", "0.5"], xlim: tuple[float, float] = None, clim: tuple[float, float] = [0, 0.001]):
        pass

class CreateOrders(ActionBase):
    CAPTION = "Create orders"

    def __call__(self, infos: list[OrderInfo]) -> OrderList:
        ol = OrderList(name="Orders")

        for name, value, disp_value in infos:
            if disp_value is None:
                disp_value = value
            ol.orders.append(OrderInfo(name, value, disp_value))

        return ol

class ExtractOrderSlicesAction(PAB):
    CAPTION = "Extract OrderSlice"

    def __call__(self, channels: list[FreqIntermediateData], orders: OrderList, tol_type: ToleranceType=ToleranceType.FixLines, tol_value: float=3) -> list[OrderSliceData]:
        order_slices = []
        for channel in channels:
            os = channel.extract_orderslice(orders, int(tol_value))
            os.name = f"OrderSlice-{channel.name}"
            order_slices.append(os)

        return order_slices

# pick up order from color plot

class ViewOrderSlice(VAB):
    CAPTION = "View OrderSlice"
    def __call__(self, order_slice: OrderSliceData):
        # switch orders
        # by reference / by frequency
        fig = self.figure
        ax = fig.gca()

        ax_orders = fig.add_axes([0.91, 0.13, 0.08, 0.2])
        ax_method = fig.add_axes([0.91, 0.01, 0.08, 0.1])
        order_labels = ["None"] + [order.name for order in order_slice.slices]
        order_choice: OrderInfo | None = None
        order_selector = RadioButtons(ax_orders, order_labels, active=0)
        method_labels = ['By frequency', 'By reference']
        method_choice = 0
        method_selector = RadioButtons(ax_method, method_labels, active=method_choice)

        def order_change(label):
            nonlocal order_choice
            for order in order_slice.slices:
                if order.name==label:
                    order_choice = order
                    break
            else:
                order_choice = None
            update()
        order_selector.on_clicked(order_change)
        def method_change(label):
            nonlocal method_choice
            method_choice = method_labels.index(label)
            update()
        method_selector.on_clicked(method_change)

        def update():
            ax.cla()
            if order_choice is None:
                fig.canvas.draw_idle()
                return
            slice = order_slice.slices[order_choice]
            if method_choice==0: # by freq
                x, y = slice.get_aligned_f()
            else:
                x, y = slice.get_aligned_ref()
            ax.plot(x, y)

            fig.canvas.draw_idle()

        self._widgets.append(method_selector)
        self._widgets.append(order_selector)

class ViewOrderSliceOfMeasurements(VAB):
    CAPTION = "View OrderSlice of measurements"
    def __call__(self, measurements: list[SimpleDefinition], orderslice_name: str):
        fig = self.figure
        ax = fig.gca()
        ax.set_title(orderslice_name)

        fiad = defaultdict(list) # fork_in_another_direction -_-,

        for measurement in measurements:
            ctx = self.container.get_context(measurement)
            orderslice: OrderSliceData = ctx.get_node_of_type(orderslice_name, OrderSliceData)
            
            for oi, sd in orderslice.slices.items():
                fiad[oi.name].append( (measurement.name, sd,) )

        ax_orders = fig.add_axes([0.91, 0.13, 0.08, 0.2])
        order_names = ["None"] + list(fiad.keys())
        order_choice:str = ""
        order_selector = RadioButtons(ax_orders, order_names, active=0)
        ax_method = fig.add_axes([0.91, 0.01, 0.08, 0.1])
        method_labels = ['By frequency', 'By reference']
        method_choice:int = 0
        method_selector = RadioButtons(ax_method, method_labels, active=method_choice)
        def order_change(label):
            nonlocal order_choice
            order_choice = label
            update()
        order_selector.on_clicked(order_change)
        def method_change(label):
            nonlocal method_choice
            method_choice = method_labels.index(label)
            update()
        method_selector.on_clicked(method_change)

        def update():
            ax.cla()
            ms = fiad.get(order_choice)
            if not ms:
                fig.canvas.draw_idle()
                return
            for (measurement_name, slice_data,) in ms:
                slice_data: SliceData
                if method_choice==0: # by freq
                    x, y = slice_data.get_aligned_f()
                else:
                    x, y = slice_data.get_aligned_ref()
                ax.plot(x, y, label=f"{measurement_name}")
            ax.legend(loc='upper right')
            fig.canvas.draw_idle()

        self._widgets.append(order_selector)
        self._widgets.append(method_selector)

# calc rms

class FilterSpectrumAction(PAB):
    CAPTION = "Filter spectrum"
    def __call__(self, channels: list[FreqDomainData], bands: list[tuple[float, float]], remove: bool=True) -> list[FreqDomainData]:
        rst = []
        if remove:
            for ch in channels:
                rst.append(ch.remove_spec(bands=bands))
        else:
            for ch in channels:
                rst.append(ch.keep_spec(bands=bands))
        return rst
    
class SpectrumToTimeAction(PAB):
    CAPTION = "Convert spectrum to TimeData"
    def __call__(self, channels: list[FreqDomainData]) -> list[TimeData]:
        rst = []
        for ch in channels:
            rst.append(ch.to_timedomain())
        return rst

class SpectrumAsTimeAction(PAB):
    CAPTION = "Treate frequency spectrum as TimeData"

class LoadCaseSpectrumComparison(VAB):
    def __call__(self, loadcases: list[str], channel_name: str):
        pass

class LoadCaseFreqIntermediateAverage(VAB):
    def __call__(self, loadcases: list[str], channel_name: str, ref_case: str):
        pass

# BearingEnvelopeAnalysis = FFT + FilterSpec + ...