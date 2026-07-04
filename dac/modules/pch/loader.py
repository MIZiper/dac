"""Loaders and process-wide cache for acquisition data files.

Provides an abstract `Loader` base class, a size-based LRU `Cache` that
is shared across all Loader instances and persists across projects, and
concrete loaders for TDMS, CSV, and HDF5 formats.
"""

import csv
import threading
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np


class Cache:
    """Process-wide, size-based LRU cache for numpy arrays.

    Cached arrays are tracked by total memory (`nbytes`). When the cache
    exceeds its configured maximum size, the least recently used entries
    are evicted and registered TimeSegments are notified to release their
    data references via `unload()`.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._max_bytes = 2 * 1024**3
        self._data: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._total_bytes = 0
        self._segments: dict[tuple, list[weakref.ref]] = {}
        self._lock = threading.Lock()

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, value: int):
        self._max_bytes = value
        with self._lock:
            self._evict_to(self._max_bytes)

    def get(self, key: tuple, read_fn=None) -> np.ndarray:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]

        if read_fn is None:
            raise KeyError(f"Cache key '{key}' not found and no read_fn provided")

        data = read_fn()
        with self._lock:
            self._put_locked(key, data)
        return data

    def _put_locked(self, key: tuple, data: np.ndarray):
        self._data[key] = data
        self._data.move_to_end(key)
        self._total_bytes += data.nbytes
        self._evict_to(self._max_bytes)

    def put(self, key: tuple, data: np.ndarray):
        with self._lock:
            self._put_locked(key, data)

    def register_segment(self, key: tuple, segment):
        with self._lock:
            self._segments.setdefault(key, []).append(weakref.ref(segment))

    def _evict_to(self, target_bytes: int):
        while self._total_bytes > target_bytes and self._data:
            key, data = self._data.popitem(last=False)
            self._total_bytes -= data.nbytes
            refs = self._segments.pop(key, [])
            for ref in refs:
                seg = ref()
                if seg is not None:
                    seg.unload()

    def clear(self):
        with self._lock:
            for key in list(self._data):
                refs = self._segments.pop(key, [])
                for ref in refs:
                    seg = ref()
                    if seg is not None:
                        seg.unload()
            self._data.clear()
            self._total_bytes = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def entry_count(self) -> int:
        return len(self._data)


class _BulkData:
    """Channel-name → array mapping with total ``nbytes`` for cache eviction.

    Keeps the Cache API compatible with bulk loaders that return
    heterogeneous arrays (different lengths / dtypes per channel).
    """

    __slots__ = ('arrays', '_nbytes')

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.arrays = arrays
        self._nbytes = sum(a.nbytes for a in arrays.values())

    @property
    def nbytes(self) -> int:
        return self._nbytes


class Loader(ABC):
    """Abstract base for data file loaders.

    Subclasses implement format-specific metadata and data reading.
    All Loader instances share the process-wide Cache via `_cache`.
    """

    _cache = Cache()

    @abstractmethod
    def load_meta(self, source: str) -> "list[TimeSegment]":
        """Read metadata from source; return TimeSegments without loading y."""
        ...

    @abstractmethod
    def _read_data(self, source: str, key: tuple) -> np.ndarray:
        """Read waveform data for a cache key from the source file."""
        ...

    def _read_bulk(self, source: str) -> _BulkData:
        """Read all channels at once for formats that cannot load
        individual channels independently (e.g. CSV).

        Returns a :class:`_BulkData` whose ``.arrays`` is a
        ``{channel_name: np.ndarray}`` mapping. Subclasses that support
        bulk loading **must** override this and :meth:`_extract_channel`;
        the default path (TDMS, HDF5) does not use bulk loading.
        """
        raise NotImplementedError

    def _extract_channel(self, bulk_data: _BulkData, source: str, key: tuple) -> np.ndarray:
        """Extract a single channel from *bulk_data* by *key*.

        *key* is the original per-channel cache key ``(source, group, name)``.
        Must be overridden together with :meth:`_read_bulk`.
        """
        raise NotImplementedError

    def load_full(self, cache_key: tuple, segment=None) -> np.ndarray:
        """Load full data array, using the cache.

        If the segment has a ``_bulk_key`` the loader reads all channels
        in one pass and caches the combined array; individual channel
        access then slices from that bulk array.  For loaders that can
        read channels independently (TDMS, HDF5) each channel is cached
        separately under its per-channel key.

        When *segment* is provided it is registered so that the cache
        can call ``segment.unload()`` on eviction.
        """
        source = cache_key[0]

        # Bulk-loading path (CSV etc.)
        bulk_key = getattr(segment, "_bulk_key", None) if segment is not None else None
        if bulk_key is not None:
            def _read_bulk_fn():
                return self._read_bulk(source)

            bulk_data = self._cache.get(bulk_key, _read_bulk_fn)
            data = self._extract_channel(bulk_data, source, cache_key)
            if segment is not None:
                self._cache.register_segment(bulk_key, segment)
            return data

        # Per-channel path (TDMS, HDF5)
        def read_fn():
            return self._read_data(source, cache_key)

        data = self._cache.get(cache_key, read_fn)
        if segment is not None:
            self._cache.register_segment(cache_key, segment)
        return data


class TDMSLoader(Loader):
    """Loader for NI TDMS (.tdms) files.

    Reads channel metadata (name, sample interval, start time, unit)
    without loading waveform data. Full data is loaded on demand.

    t0 is returned as ``np.datetime64`` when the file contains a valid
    ``wf_start_time`` property.
    """

    def load_meta(self, source: str) -> "list[TimeSegment]":
        from nptdms import TdmsFile
        from . import TimeSegment

        f = TdmsFile.read_metadata(source)
        segments = []
        for group in f.groups():
            for channel in group.channels():
                props = channel.properties
                t0 = self._tdms_time_to_datetime64(props.get("wf_start_time"))
                dt = float(props.get("wf_increment", 1.0))
                n_samples = int(props.get("wf_samples", 0))
                cache_key = (source, group.name, channel.name)
                seg = TimeSegment(
                    name=channel.name,
                    t0=t0,
                    length=n_samples,
                    dt=dt,
                    y_unit=str(
                        props.get("unit_string", props.get("Unit", "-"))
                    ),
                    comment=str(
                        props.get(
                            "description", props.get("Description", "")
                        )
                    ),
                    _cache_key=cache_key,
                    _loader=self,
                )
                segments.append(seg)
        return segments

    def _read_data(self, source: str, key: tuple) -> np.ndarray:
        from nptdms import TdmsFile

        _, group_name, channel_name = key
        f = TdmsFile(source, read_metadata_only=False)
        for group in f.groups():
            if group.name == group_name:
                for channel in group.channels():
                    if channel.name == channel_name:
                        return channel.data
        raise ValueError(
            f"Channel '{channel_name}' not found in group '{group_name}'"
        )

    @staticmethod
    def _tdms_time_to_datetime64(wf_start_time) -> np.datetime64:
        if wf_start_time is None:
            return np.datetime64("NaT")
        from datetime import datetime

        if isinstance(wf_start_time, np.datetime64):
            return wf_start_time
        if isinstance(wf_start_time, datetime):
            return np.datetime64(wf_start_time.replace(tzinfo=None).isoformat())
        return np.datetime64("NaT")


class CSVLoader(Loader):
    """Loader for CSV files — reads the entire file once and caches the
    combined array, then extracts individual columns on request.

    Expects CSV with a header row. The first column is treated as the
    time index when ``has_time_column`` is True. Remaining columns become
    TimeSegments. Delimiter and skiprows can be configured per source.
    """

    def __init__(self):
        super().__init__()
        self._formats: dict[str, dict] = {}
        self._col_index: dict[str, dict[str, int]] = {}

    def load_meta(
        self,
        source: str,
        delimiter: str = ",",
        skiprows: int = 0,
        has_time_column: bool = True,
        dt: float = 1.0,
        t0: "float | np.datetime64" = 0.0,
    ) -> "list[TimeSegment]":
        from . import TimeSegment

        self._formats[source] = {
            "delimiter": delimiter,
            "skiprows": skiprows,
            "has_time_column": has_time_column,
        }

        with open(source, "r") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for _ in range(skiprows):
                next(reader)
            headers = next(reader)

        n_lines = 0
        with open(source, "r") as f:
            n_lines = sum(1 for _ in f) - skiprows - 1

        start_col = 1 if has_time_column else 0
        col_names = [headers[i].strip() or f"col_{i}"
                     for i in range(start_col, len(headers))]
        self._col_index[source] = {name: idx
                                   for idx, name in enumerate(col_names)}

        bulk_key = (source,)
        segments = []
        for idx, name in enumerate(col_names):
            cache_key = (source, "", name)
            seg = TimeSegment(
                name=name,
                t0=t0,
                length=max(0, n_lines),
                dt=dt,
                y_unit="-",
                comment="",
                _cache_key=cache_key,
                _bulk_key=bulk_key,
                _loader=self,
            )
            segments.append(seg)
        return segments

    def _read_bulk(self, source: str) -> _BulkData:
        fmt = self._formats.get(source, {})
        delimiter = fmt.get("delimiter", ",")
        skiprows = fmt.get("skiprows", 0)
        has_time_column = fmt.get("has_time_column", True)

        usecols = None
        if has_time_column:
            n_cols = 1 + len(self._col_index[source])
            usecols = list(range(1, n_cols))

        data = np.loadtxt(
            source,
            delimiter=delimiter,
            skiprows=skiprows + 1,
            usecols=usecols,
            ndmin=2,
        )
        arrays = {}
        for name, idx in self._col_index[source].items():
            arrays[name] = data[:, idx]
        return _BulkData(arrays)

    def _extract_channel(self, bulk_data: _BulkData, source: str, key: tuple) -> np.ndarray:
        _, _, name = key
        return bulk_data.arrays[name]

    def _read_data(self, source: str, key: tuple) -> np.ndarray:
        raise NotImplementedError(
            "CSVLoader uses bulk loading; _read_data should not be called directly"
        )


class HDF5Loader(Loader):
    """Loader for HDF5 (.h5 / .hdf5) files. Requires h5py."""

    def load_meta(self, source: str) -> "list[TimeSegment]":
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 loading. "
                "Install with: pip install h5py"
            )

        from . import TimeSegment

        segments = []
        with h5py.File(source, "r") as f:
            for name, dataset in self._iter_datasets(f, h5py):
                shape = dataset.shape
                length = shape[0] if len(shape) >= 1 else 0

                cache_key = (source, "", name)
                seg = TimeSegment(
                    name=name,
                    t0=0.0,
                    length=length,
                    dt=1.0,
                    y_unit="-",
                    comment="",
                    _cache_key=cache_key,
                    _loader=self,
                )
                segments.append(seg)
        return segments

    def _read_data(self, source: str, key: tuple) -> np.ndarray:
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 loading. "
                "Install with: pip install h5py"
            )

        _, _, name = key
        with h5py.File(source, "r") as f:
            return f[name][()]

    @staticmethod
    def _iter_datasets(h5file, h5py, prefix: str = ""):
        for name, item in h5file.items():
            full = f"{prefix}/{name}" if prefix else name
            if isinstance(item, h5py.Dataset):
                yield full, item
            elif isinstance(item, h5py.Group):
                yield from HDF5Loader._iter_datasets(item, h5py, full)
