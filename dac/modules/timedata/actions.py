import numpy as np
import re
from scipy import signal

from dac.core.actions import ActionBase, VAB, PAB, SAB
from . import TimeData
from .data_loader import load_tdms
from ..nvh import FilterType

class LoadAction(PAB):
    CAPTION = "Load measurement data"
    def __call__(self, fpaths: list[str], ftype: str=None) -> list[TimeData]:
        """Loads time-series data from files.

        Currently supports TDMS files. Iterates through a list of file paths,
        loads TDMS data using `load_tdms` from `.data_loader`.

        Parameters
        ----------
        fpaths : list[str]
            A list of strings representing file paths to load.
        ftype : str, optional
            File type string (currently unused, but could be used to
            select different loaders), by default None.

        Returns
        -------
        list[TimeData]
            A list of TimeData objects loaded from the files.
        """

        n = len(fpaths)
        rst = []
        for i, fpath in enumerate(fpaths):
            if not fpath.upper().endswith("TDMS"):
                continue
            r = load_tdms(fpath=fpath)
            rst.extend(r)
            self.progress(i+1, n)
        return rst

class TruncAction(ActionBase):
    CAPTION = "Truncate TimeData"
    def __call__(self, channels: list[TimeData], duration: tuple[float, float]=[0, 0]) -> list[TimeData]:
        """Truncates TimeData channels to a specified duration.

        Parameters
        ----------
        channels : list[TimeData]
            A list of TimeData objects to truncate.
        duration : (start_time, end_time)
            Units in seconds.
            If end_time is 0, truncates to the end of the signal.
            If end_time is negative, it's an offset from the end.

        Returns
        -------
        list[TimeData]
            A list of new TimeData objects, each truncated to the specified
            duration. The names of the new channels are appended with "-Trunc".
        """

        rst = []
        xfrom, xto = duration

        for i, channel in enumerate(channels):
            x = channel.x
            if xto==0:
                idx_to = None
            else:
                if xto<0:
                    xto = x[-1] + xto
                idx_to = np.searchsorted(x, xto)

            idx_from = np.searchsorted(x, xfrom)
            y = channel.y[idx_from:idx_to]
            rst.append(TimeData(f"{channel.name}-Trunc", y=y, dt=channel.dt, y_unit=channel.y_unit, comment=channel.comment))
        
        return rst

class FilterAction(ActionBase):
    CAPTION = "Filter TimeData"
    def __call__(self, channels: list[TimeData], freqs: tuple[float, float], order: int=3, filter_type: FilterType=FilterType.BandPass) -> list[TimeData]:
        """Smooth `TimeData` with `butter` filter.

        Parameters
        ----------
        channels : list[TimeData]
            Input channels, list of original `TimeData`
        freqs : (start_frequency, end_frequency)
        order : int
        filter_type : 'LowPass' | 'HighPass' | 'BandPass' | 'BandStop'

        Returns
        -------
        list[TimeData]
            A list of filtered `TimeData`
        """

        # with current annotation mechanism, freq as single float won't be passed inside
        rst = []

        if filter_type in (FilterType.BandPass, FilterType.BandStop):
            w = np.array(freqs)
        else:
            w = freqs[0]

        for i, channel in enumerate(channels):
            Wn = w / (channel.fs / 2)
            b, a = signal.butter(order, Wn, filter_type.value)
            y = signal.filtfilt(b, a, channel.y)

            rst.append(TimeData(name=f"{channel.name}-FiltT", y=y, dt=channel.dt, y_unit=channel.y_unit, comment=channel.comment))

        return rst

class ResampleAction(ActionBase):
    CAPTION = "Resample data to"
    def __call__(self, channels: list[TimeData], dt: float=1) -> list[TimeData]:
        """Resamples TimeData channels to a new time interval (dt).

        Downsamples by taking every Nth point, where N is `dt / original_dt`.
        If the new dt is smaller than or equal to the original dt,
        the original channel is returned.

        Parameters
        ----------
        channels : list[TimeData]
            A list of TimeData objects to resample.
        dt : float, default 1
            The desired new time interval in seconds.

        Returns
        -------
        list[TimeData]
            A list of TimeData objects, resampled to the new dt.
        """

        rst = []
        for i, channel in enumerate(channels):
            interval = int(dt // channel.dt)
            if interval > 1:
                rst.append(TimeData(name=channel.name, y=channel.y[::interval], dt=channel.dt*interval, y_unit=channel.y_unit, comment=channel.comment))
            else:
                rst.append(channel)
        return rst

class PrepDataAction(SAB, seq=[TruncAction, ResampleAction, FilterAction]): # example sequences
    ...

class EnvelopeTimeAction(PAB):
    CAPTION = "Envelop with Hilbert transform"
    # only works when there is positive and negative part
    def __call__(self, channels: list[TimeData], restore_offset: bool=False) -> list[TimeData]:
        """Calculates the envelope of TimeData channels using the Hilbert transform.

        The mean of the signal (offset) is removed before applying the Hilbert
        transform, and the envelope is calculated as the absolute value of the
        analytic signal.

        Parameters
        ----------
        channels : list[TimeData]
            A list of TimeData objects.
        restore_offset : bool, default False
            If True, the original offset is added back to the envelope.

        Returns
        -------
        list[TimeData]
            A list of new TimeData objects representing the envelope of each
            input channel. The names are appended with "-Env".
        """

        rst = []
        for i, channel in enumerate(channels):
            channel_y = channel.y
            offset = np.mean(channel_y)
            analytic = signal.hilbert(channel_y-offset)
            env = np.abs(analytic)
            if restore_offset:
                env += offset

            rst.append(TimeData(name=f"{channel.name}-Env", y=env, dt=channel.dt, y_unit=channel.y_unit, comment=channel.comment))

        return rst

class ShowTimeDataAction(VAB):
    CAPTION = "Show measurement data"
    def __call__(self, channels: list[TimeData], plot_dt: float=None, xlim: tuple[float, float]=None, ylim: tuple[float, float]=None):
        """Displays TimeData channels on a Matplotlib figure.

        Plots each channel on the same axes. Allows for optional downsampling
        for plotting and setting x/y axis limits.

        Parameters
        ----------
        channels : list[TimeData]
            A list of TimeData objects to plot.
        plot_dt : float, optional
            If provided, resamples the data for plotting to this
            time interval to speed up rendering, by default None (no resampling).
        xlim : tuple[float, float], optional
            Tuple (min, max) for the x-axis (time) limits, by default None.
        ylim : tuple[float, float], optional
            Tuple (min, max) for the y-axis (amplitude) limits, by default None.
        """

        fig = self.figure
        fig.suptitle("Time data visualization")

        ax = fig.gca()
        ax.set_xlabel("Time [s]")
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        
        for channel in channels:
            x, y = channel.x, channel.y
            if plot_dt is not None:
                interval = int(plot_dt // channel.dt)
                if interval > 1:
                    x = x[::interval]
                    y = y[::interval]
            
            ax.plot(x, y, label=f"{channel.name} [{channel.y_unit}]")
        
        ax.legend(loc="upper right")

class CounterToTachoAction(ActionBase):
    CAPTION = "Encoder counter to tacho"
    def __call__(self, channel: TimeData, ppr: int=1024, sr_delta: float=0.1) -> TimeData:
        """Converts an encoder counter signal (TimeData) to a tachometer (RPM) signal.

        Calculates RPM based on the difference in counter values over a
        time window defined by `sr_delta` (sampling rate delta).

        Parameters
        ----------
        channel : TimeData
            TimeData representing the encoder counter values.
        ppr : int, default 1024
            Pulses per revolution of the encoder.
        sr_delta : float, default 0.1
            A ratio to sample rate,
            and determine the time window to calculate the speed.

        Returns
        -------
        TimeData
            A new TimeData object representing the calculated RPM, named "Tacho".
        """

        counter = channel.y
        delta = int(channel.fs * sr_delta)
        rpm = np.zeros(len(counter))
        rpm[delta:-delta] = (counter[2*delta:]-counter[:-2*delta]) / ppr / (2*delta*channel.dt / 60)
        rpm[:delta] = rpm[delta]
        rpm[-delta:] = rpm[-delta-1]

        return TimeData(name="Tacho", y=rpm, dt=channel.dt, y_unit="rpm")
    
class ChopOffSpikesAction(PAB):
    CAPTION = "Chop off spikes"
    def __call__(self, channels: list[TimeData], max_dydt: float, limits: tuple[float, float]=None) -> list[TimeData]:
        """Removes spikes from TimeData channels by limiting the rate of change.

        Iterates through the data and if a segment's rate of change (`dy/dt`)
        exceeds `max_dydt`, it interpolates linearly between the start and
        end points of that "spike" segment.

        Parameters
        ----------
        channels : list[TimeData]
            A list of TimeData objects to process.
        max_dydt : float
            The maximum allowed change in y per second.
        limits : tuple[float, float], optional
            Optional tuple (min_val, max_val) to clip the output values.
            If None, no clipping is applied beyond spike removal, by default None.

        Returns
        -------
        list[TimeData]
            A list of new TimeData objects with spikes removed/interpolated.
            The names are appended with "-Chop".
        """
        # `max_dydt`: maximum delta y per sec

        if limits is None:
            low_lim, high_lim = -np.inf, np.inf
        else:
            low_lim, high_lim = limits

        ret_channels = []
        n = len(channels)

        for j, channel in enumerate(channels):
            signal = channel
            orig_data = signal.y
            target_data = orig_data.copy()

            max_dy = max_dydt * signal.dt

            look4raise = True
            s, e = 0, 0

            if len(target_data)>0:
                pv = orig_data[0]
            else:
                continue
            
            for i, cv in enumerate(orig_data[1:]):
                dy = cv-pv
                if look4raise:
                    if dy > max_dy:
                        s = i # start from prev idx
                        sv = pv
                        look4raise = False
                else:
                    if np.abs(dy)<max_dy and np.abs(cv-sv)<max_dy*(i+1-s) and cv<=high_lim and cv>=low_lim:
                        look4raise = True
                        target_data[(s+1):(i+1)] = np.linspace(sv, cv, num=(i-s+2), endpoint=True)[1:-1]
                pv = cv
            
            ret_channels.append(
                TimeData(
                    name=f"{channel.name}-Chop",
                    y=target_data,
                    dt=signal.dt,
                    y_unit=signal.y_unit
                )
            )

            self.progress(j+1, n)

        return ret_channels

class PulseToAzimuthAction(ActionBase):
    CAPTION = "Pulse to azimuth"
    def __call__(self, channel: TimeData, ref_level: float, ppr: int=1, higher_as_pulse: bool=True, phase_shift: float=0, ref_channel: TimeData=None) -> TimeData:
        """Converts a pulse channel (TimeData) to an azimuth (angle) channel.

        Detects pulses based on `ref_level`. If `ref_channel` is provided,
        the azimuth is calculated considering the (potentially variable) speed
        from `ref_channel`. Otherwise, a linear azimuth is assumed between pulses.

        Parameters
        ----------
        channel : TimeData
            TimeData representing the pulse signal.
        ref_level : float
            The threshold level to detect pulses.
        ppr : int, default 1
            Pulses per revolution. Used to scale the final angle to 0-360 degrees.
        higher_as_pulse : bool, optional
            If True (default), values above `ref_level` are
            considered pulses. If False, values below are pulses.
        phase_shift : float, optional
            A phase shift in degrees to add to the final azimuth, by default 0.
        ref_channel : TimeData, optional
            Optional TimeData representing a reference speed signal.
            If provided, its values are used to calculate a non-linear
            azimuth between pulses, by default None.

        Returns
        -------
        TimeData
            A new TimeData object representing the azimuth in degrees (0-360).
            The name is "Azi-{channel.name}".
        """

        # `ref_channel` is for example when speed ramping, azimuth is not linear equal
        if ref_channel:
            sr_ratio = ref_channel.dt / channel.dt
            ref_data = ref_channel.y

        data = channel.y
        inpulse = data>ref_level if higher_as_pulse else data<ref_level
        indexes = np.arange(len(data))[inpulse]
        idx_diff = np.diff(indexes)
        # assert len(idx_diff) == len(indexes)-1
        idx_pulse_end = indexes[:-1][idx_diff>np.mean(idx_diff)]
        ang_data = np.zeros_like(data) * np.nan
        for i, (from_idx, to_idx) in enumerate(zip(idx_pulse_end[:-1], idx_pulse_end[1:])):
            if ref_channel:
                ref_indexes = (np.arange(from_idx, to_idx) / sr_ratio).astype(int)
                aligned_subdata = ref_data[ref_indexes]
                ang_data[from_idx:to_idx] = np.cumsum(aligned_subdata) / np.sum(aligned_subdata) * 360 + 360*i
            else:
                ang_data[from_idx:to_idx] = np.arange(to_idx-from_idx)/(to_idx-from_idx)*360 + 360*i
        
        return TimeData(
            name=f"Azi-{channel.name}", dt=channel.dt, y_unit="°",
            y=(ang_data/ppr+phase_shift)%360,
        )
    
class RefPulseToAzimuthAction(PAB):
    ... # create azimuth using reference

class OpAction(ActionBase):
    CAPTION = "Operation on TimeData"
    def __call__(self, channels: list[TimeData], op_str: str="{0}", y_unit: str="-") -> TimeData:
        """Performs a specified operation on a list of TimeData channels.

        The operation is defined by `op_str`, which can be a Python expression
        using "{i}" to refer to the i-th channel in the `channels` list.
        All input channels are assumed to have the same `dt` and length.

        Parameters
        ----------
        channels : list[TimeData]
            A list of TimeData objects.
        op_str : str
            A string defining the operation. Placeholders like "{0}",
            "{1}" will be replaced by `ys[0]`, `ys[1]` respectively,
            where `ys` is a list of the y-arrays of the channels.
            Example: "{0} + {1} * 2".
        y_unit : str, optional
            The unit for the resulting TimeData, by default "-".

        Returns
        -------
        TimeData
            A new TimeData object named "Channel-Op_ed" containing the result
            of the operation.
        """
        
        # assert same dt and same length
        ys = [channel.y for channel in channels]
        y = eval(re.sub(r"{(\d+)}", r"ys[\1]", op_str))
        return TimeData("Channel-Op_ed", y=y, dt=channels[0].dt, y_unit=y_unit)