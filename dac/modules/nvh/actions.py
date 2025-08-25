"""Actions for NVH (Noise, Vibration, and Harshness) analysis in the DAC framework.
"""

import numpy as np
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import TextBox, RadioButtons
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent

from dac.core.data import SimpleDefinition
from dac.core.actions import ActionBase, VAB, PAB, SAB
from dac.modules.timedata import TimeData
from . import WindowType, BandCorrection, BinMethod, AverageType, ToleranceType
from .data import FreqIntermediateData, DataBins, FreqDomainData, \
                  OrderInfo, OrderList, SliceData, OrderSliceData

class ToFreqDomainAction(PAB):
    CAPTION = "Simple FFT to frequency domain" # rect window

    def __call__(self, channels: list[TimeData], window: WindowType=WindowType.Uniform, corr: BandCorrection=BandCorrection.NarrowBand) -> list[FreqDomainData]:
        """Converts TimeData channels to the frequency domain using FFT.

        A simple conversion without advanced windowing or overlapping options.
        Primarily uses a uniform window unless Hanning or Hamming is specified.

        Parameters
        ----------
        channels : list[TimeData]
            A list of TimeData objects to convert.
        window : WindowType, default WindowType.Uniform
            The windowing function to apply, available:
            - Uniform
            - Hanning
            - Hamming
        corr : BandCorrection, default BandCorrection.NarrowBand
            Band correction type, available:
            - NarrowBand
            - BroadBand

        Returns
        -------
        list[FreqDomainData]
            A list of FreqDomainData objects, each representing the spectrum
            of the corresponding input channel.
        """
        
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
        """Converts TimeData channels to FreqIntermediateData using STFT-like processing.

        This method processes channels in segments (bins) with specified
        resolution and overlap, applying a window function. If a reference
        channel is provided, its mean value per segment is used for the
        reference bins; otherwise, time is used.

        Parameters
        ----------
        channels : list[TimeData]
            A list of TimeData objects to convert.
        window : 'Hanning' | 'Hamming'
            The windowing function to apply.
        corr : BandCorrection
            Band correction type, by default BandCorrection.NarrowBand.
        resolution : float, default 0.5Hz
            Desired frequency resolution (df) in Hz for the STFT.
        overlap : float
            Overlap ratio between consecutive segments (0.0 to 1.0), by default 0.75.
        ref_channel : TimeData, optional
            Optional TimeData to use for creating reference bins.
            If None, time progression is used as reference, by default None.

        Returns
        -------
        list[FreqIntermediateData]
            A list of FreqIntermediateData objects.
        """
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
        """Averages FreqIntermediateData channels to produce FreqDomainData (spectra).

        Parameters
        ----------
        channels : list[FreqIntermediateData]
            A list of FreqIntermediateData objects to average.
        average_by : 'Energy' | 'Linear'
            The averaging method to use.

        Returns
        -------
        list[FreqDomainData]
            A list of FreqDomainData objects, each representing the averaged
            spectrum of the corresponding input FreqIntermediateData.
        """
        
        rst = []
        for channel in channels:
            rst.append(channel.to_powerspectrum(average_by=average_by))
        return rst
    
class ViewFreqDomainAction(VAB):
    CAPTION = "Show FFT spectrum"

    def __call__(self, channels: list[FreqDomainData], xlim: tuple[float, float]=None, ylim: tuple[float, float]=None, with_phase: bool=False):
        """Displays FreqDomainData (spectra) on a Matplotlib figure.

        Plots the amplitude spectrum for each channel. Optionally, can also
        display the phase spectrum on a separate subplot.

        Parameters
        ----------
        channels : list[FreqDomainData]
            A list of FreqDomainData objects to plot.
        xlim : tuple[float, float], optional
            Tuple (min, max) for the x-axis (frequency) limits.
        ylim : tuple[float, float], optional
            Tuple (min, max) for the y-axis (amplitude) limits.
        with_phase : bool, default False
            If True, plots the phase spectrum below the amplitude
            spectrum.
        """
        
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
        """Displays FreqIntermediateData as a colormap (spectrogram-like) plot.

        The x-axis represents frequency, the y-axis represents the reference
        bins (e.g., time or RPM), and the color intensity represents amplitude.

        Parameters
        ----------
        channel : FreqIntermediateData
            The FreqIntermediateData object to plot.
        xlim : tuple[float, float], optional
            Tuple (min, max) for the x-axis (frequency) limits.
        clim : tuple[float, float], default [0, 0.001]
            Tuple (min, max) for the color axis limits.
        """
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
        """Extracts peak amplitudes from FreqDomainData at specified frequencies.

        For each specified frequency, it finds the peak amplitude within a given
        tolerance (number of frequency lines).

        Parameters
        ----------
        channels : list[FreqDomainData]
            A list of FreqDomainData objects to process.
        frequencies : list[float]
            A list of frequencies at which to extract amplitudes.
        line_tol : int, default 3
            Number of frequency lines around the target frequency
            to search for the peak.

        Returns
        -------
        list
            A list of lists, where each inner list contains tuples of
            (actual_frequency, peak_amplitude) for each specified frequency
            in the corresponding channel. Structure:

                [[ (f1_ch1, amp1_ch1), (f2_ch1, amp2_ch1), ... ],  # Channel 1
                [ (f1_ch2, amp1_ch2), (f2_ch2, amp2_ch2), ... ]]  # Channel 2

            (Note: The actual implementation is missing, currently returns `...`)
        """
        ...

class ViewColorPlotAndCheckOrderSlice(ViewFreqIntermediateAction):
    # obsolete, it's too laggy
    CAPTION = "Show color plot with order indication"
    def __call__(self, channel: FreqIntermediateData, xlim: tuple[float, float] = None, clim: tuple[float, float] = [0, 0.001]):
        """Displays a FreqIntermediateData colormap and allows interactive order checking.

        (Obsolete: This class is marked as too laggy in the source code comments)

        Inherits from ViewFreqIntermediateAction. Adds a TextBox for users to input
        an order value. An order line is then drawn on the colormap.
        Or by clicking on the plot to get order number.
        """

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

class MarkOrders(VAB): # Mark orders for colorplot, I suppose?
    pass

class ViewColorPlotWithOrderSlice(ViewFreqIntermediateAction): # inherit SAB = ViewColorPlot + MarkOrders
    CAPTION = "Show color plot with order indication"
    def __call__(self, channel: FreqIntermediateData, orders: OrderList, fmt_lines: list[str]=["{f_1}", "0.5"], xlim: tuple[float, float] = None, clim: tuple[float, float] = [0, 0.001]):
        """Displays a FreqIntermediateData colormap with order lines.

        (Note: The actual implementation is missing, currently a `pass` statement)

        Parameters
        ----------
        channel : FreqIntermediateData
            The FreqIntermediateData object to plot.
        orders : OrderList
            An OrderList defining the orders to mark.
        fmt_lines : list[str]
            A list of strings for custom frequency lines (similar to
            ShowFreqLinesFreq).
        xlim : tuple[float, float], optional
            Tuple (min, max) for the x-axis (frequency) limits.
        clim : tuple[float, float], optional
            Tuple (min, max) for the color axis limits.
        """
        pass

class CreateOrders(ActionBase):
    CAPTION = "Create orders"

    def __call__(self, infos: list[OrderInfo]) -> OrderList:
        """Creates an OrderList from a list of OrderInfo tuples or parameters.

        Parameters
        ----------
        infos : list[OrderInfo]
            A list where each item defines an order.
            Each item should be a tuple (name, value).
            `name` is the label (e.g., "f_1").
            `value` is the order, normally reference to frequency.

        Returns
        -------
        OrderList
            An OrderList object containing the defined orders.
        """
        ol = OrderList(name="Orders")

        for name, value in infos:
            ol.orders.append(OrderInfo(name, value))

        return ol
    
class GuessOrdersOnSpectrum(VAB):
    CAPTION = "Guess orders on spectrum graph"
    def __call__(self, ol: OrderList, speed: TimeData, tol: float=0.2, max_harm: int=3):
        """Guess closest orders (characteristic frequencies) based on selection on spectrum graph.

        Use left button to select a range in spectrum graph, and right button to cyclic the harmonic number.
        A list is shown for the closest orders.
        
        Parameters
        ----------
        ol : OrderList
        tol : float
            Tolerance for the deviations, a ratio between 0-1.
        max_harm : int, default 3
            Max harmonics for searching when cyclic-ing.
        """

        fig = self.figure
        ax = fig.gca()
        canvas = fig.canvas

        speed = speed.y.mean()

        cur_harm = 1
        MODE_NONE, MODE_RANGE, MODE_SINGLE = range(3)
        mode = MODE_NONE
        x0 = None
        x1 = None
        table_x = 0.0
        table_y = 0.0

        span = ax.axvspan(0, 0, color='red', alpha=0.5, visible=False)
        table = None

        def clear_table():
            nonlocal table
            if table is not None:
                table.remove()
                table = None
                canvas.draw_idle()

        def show_table(x0, x1, harm, table_x, table_y):
            nonlocal table
            if table is not None:
                table.remove()
            f_sel = abs(x1-x0)
            rows = []
            for order in ol.orders:
                f_order = order.value * harm * speed/60
                deviation = abs(f_sel - f_order) / f_order if f_order != 0 else 0
                if deviation <= tol:
                    rows.append([
                        f"{harm}x {order.name}",
                        f"{f_order:.3f}",
                        f"{deviation*100:.1f}%"
                    ])
            if not rows:
                rows = [[f"{harm}x -", "-", "-"]]
            # Table position: left bottom, starting from mouse click
            ax_xlim = ax.get_xlim()
            ax_ylim = ax.get_ylim()
            x_frac = (table_x - ax_xlim[0]) / (ax_xlim[1] - ax_xlim[0])
            y_frac = (table_y - ax_ylim[0]) / (ax_ylim[1] - ax_ylim[0])
            x_frac = max(0, min(1, x_frac))
            y_frac = max(0, min(1, y_frac))
            table = ax.table(
                rows,
                colLabels=['Name', 'Value', 'Devi.'],
                colWidths=[1.5, 0.7, 0.5],
                cellLoc='right',
                loc='center',
                bbox=[x_frac, y_frac, 0.25, 0.05*max(len(rows), 1)],
            )
            canvas.draw_idle()

        def event_start(e: MouseEvent):
            if canvas.widgetlock.locked():
                return
            nonlocal mode, x0, x1, cur_harm, table_x, table_y
            if e.button == MouseButton.LEFT and mode==MODE_NONE:
                mode = MODE_RANGE
                x0 = e.xdata
                x1 = e.xdata
                table_x = e.xdata if e.xdata is not None else 0.0
                table_y = e.ydata if e.ydata is not None else 0.0
                span.set_x(x0)
                span.set_width(0)
                span.set_visible(True)
                cur_harm = 1
                clear_table()

            elif e.button == MouseButton.MIDDLE and mode==MODE_NONE:
                if e.xdata is not None:
                    mode = MODE_SINGLE
                    x0 = 0
                    x1 = e.xdata
                    table_x = e.xdata
                    table_y = e.ydata if e.ydata is not None else 0.0
                    cur_harm = 1
                    show_table(x0, x1, cur_harm, table_x, table_y)

        def event_end(e: MouseEvent):
            if canvas.widgetlock.locked():
                return
            nonlocal mode, x0, x1
            if e.button == MouseButton.LEFT and mode==MODE_RANGE:
                mode = MODE_NONE
                x0 = None
                x1 = None
                span.set_visible(False)
                clear_table()
                
            elif e.button == MouseButton.MIDDLE and mode==MODE_SINGLE:
                mode = MODE_NONE
                x0 = None
                x1 = None
                clear_table()

        def toggle_harmonic(e: MouseEvent):
            if canvas.widgetlock.locked():
                return
            nonlocal cur_harm
            if e.button == MouseButton.RIGHT and mode!=MODE_NONE:
                cur_harm = (cur_harm % max_harm) + 1
                show_table(x0, x1, cur_harm, table_x, table_y)

        def motion(e: MouseEvent):
            if canvas.widgetlock.locked():
                return
            nonlocal x1, table_x, table_y
            if mode==MODE_RANGE and e.xdata is not None:
                x1 = e.xdata
                table_x = e.xdata if e.xdata is not None else 0.0
                table_y = e.ydata if e.ydata is not None else 0.0
                span.set_width(x1-x0)
                show_table(x0, x1, cur_harm, table_x, table_y)

        cid1 = canvas.mpl_connect('button_press_event', event_start)
        cid2 = canvas.mpl_connect('button_release_event', event_end)
        cid3 = canvas.mpl_connect('button_press_event', toggle_harmonic)
        cid4 = canvas.mpl_connect('motion_notify_event', motion)

        self._cids.extend([cid1, cid2, cid3, cid4])
        
class ShowSpectrumWithGuesser(SAB, seq=[ViewFreqDomainAction, GuessOrdersOnSpectrum]):
    CAPTION = "Show spectrum with order guesser"

class ExtractOrderSlicesAction(PAB):
    CAPTION = "Extract OrderSlice"

    def __call__(self, channels: list[FreqIntermediateData], orders: OrderList, tol_type: ToleranceType=ToleranceType.FixLines, tol_value: float=3) -> list[OrderSliceData]:
        """Extracts order slices from FreqIntermediateData channels.

        For each channel and each specified order, it extracts the amplitude
        along the order line in the frequency-reference plane.

        Parameters
        ----------
        channels : list[FreqIntermediateData]
            A list of FreqIntermediateData objects.
        orders : OrderList
            An OrderList defining the orders to extract.
        tol_type : ToleranceType
            Tolerance type for order extraction, by default ToleranceType.FixLines.
            (Note: only tol_value is used as int(tol_value) for line_tol,
            tol_type itself is not used in the current implementation).
        tol_value : float, default 3
            Tolerance value. When tol_type is FixLines, this is
            interpreted as the number of frequency lines around the
            order line to search for the peak.

        Returns
        -------
        list[OrderSliceData]
            A list of OrderSliceData objects, one for each input channel.
        """
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
        """Displays OrderSliceData, allowing selection of different orders and views.

        Provides radio buttons to select an order from the OrderSliceData and
        to switch the x-axis between 'By frequency' and 'By reference'.

        Parameters
        ----------
        order_slice : OrderSliceData
            The OrderSliceData object to display.
        """
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
        """Displays a comparison of a specific order slice across multiple measurements.

        Each "measurement" is a SimpleDefinition context key. The action retrieves
        an OrderSliceData object (specified by `orderslice_name`) from the
        context of each measurement and plots the selected order.

        Parameters
        ----------
        measurements : list[SimpleDefinition]
            A list of SimpleDefinition objects, each representing
            a measurement context.
        orderslice_name : str
            The name of the OrderSliceData node to retrieve
            from each measurement context.
        """
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
        """Filters FreqDomainData channels by removing or keeping specified frequency bands.

        Parameters
        ----------
        channels : list[FreqDomainData]
            A list of FreqDomainData objects to filter.
        bands : list of [f_from, f_to]
            A list of tuples, where each tuple (ffrom, fto) defines a
            frequency band.
        remove : bool
            If True (default), frequencies within the specified bands are
            zeroed out. If False, only frequencies within the specified
            bands are kept, and everything else is zeroed out.

        Returns
        -------
        list[FreqDomainData]
            A list of new FreqDomainData objects with the filter applied.
        """
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
        """Converts FreqDomainData (spectra) back to the time domain using IFFT.

        Parameters
        ----------
        channels : list[FreqDomainData]
            A list of FreqDomainData objects.

        Returns
        -------
        list[TimeData]
            A list of TimeData objects, each representing the time-domain
            signal of the corresponding input spectrum.
        """
        rst = []
        for ch in channels:
            rst.append(ch.to_timedomain())
        return rst

class SpectrumAsTimeAction(PAB):
    CAPTION = "Treat frequency spectrum as TimeData"

class LoadCaseSpectrumComparison(VAB):
    CAPTION = "Show spectrum across loadcases"
    def __call__(self, loadcases: list[SimpleDefinition], channel_name: str):
        """Compares spectra of a specific channel across multiple load cases.

        Parameters
        ----------
        loadcases : list[SimpleDefinition]
            A list of different load cases (contexts).
        channel_name : str
            The name of the FreqDomainData channel to compare.
        """
        fig = self.figure
        ax = fig.gca()
        ax.set_title(channel_name)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude")

        for loadcase in loadcases:
            ctx = self.container.get_context(loadcase)
            freqdata: FreqDomainData = ctx.get_node_of_type(channel_name, FreqDomainData)
            ax.plot(freqdata.x, freqdata.amplitude, label=f"{loadcase.name} [{freqdata.y_unit}]")

        ax.legend(loc="upper right")

class LoadCaseFreqIntermediateAverage(VAB):
    def __call__(self, loadcases: list[str], channel_name: str, ref_case: str):
        """Averages FreqIntermediateData for a channel across multiple load cases,
        potentially referencing one specific case.

        (Note: The actual implementation is missing, currently a `pass` statement)

        Parameters
        ----------
        loadcases : list[str]
            A list of strings for load case identifiers.
        channel_name : str
            The name of the FreqIntermediateData channel.
        ref_case : str
            A string identifying the reference load case.
        """
        pass

# BearingEnvelopeAnalysis = FFT + FilterSpec + ...