"""Process-level file cache for lazy data loading.

``FileStore`` is a singleton that caches opened measurement files so that
multiple actions (or even multiple ``Container`` instances in desktop
mode) can share file handles and avoid repeated I/O.

Files are opened in metadata-only mode — channel metadata (name, dt,
unit, sample count) is read instantly while the actual data arrays are
only loaded on first access via ``LazyTimeData``.

The cache uses LRU eviction with a configurable limit on the number of
concurrently open files.  This keeps file-descriptor usage bounded
without needing a complicated reference-counting scheme.

Extension point
---------------
To support new file formats, register a *meta-loader* and an
*open function*::

    FileStore.register_meta_loader(".csv", my_csv_meta_loader)
    FileStore.set_open_fn(my_open_fn)
"""

import threading
from typing import Any, Callable

from nptdms import TdmsFile


def _default_open(fpath: str) -> TdmsFile:
    """Default file opener — metadata-only TDMS."""
    return TdmsFile(fpath, read_metadata_only=True, keep_open=True)


class FileStore:
    """LRU cache of opened measurement files, shared process-wide.

    Thread-safe.  Singleton via ``FileStore.instance()``.
    """

    _instance: "FileStore | None" = None

    def __init__(self, max_open_files: int = 50):
        self._cache: dict[str, Any] = {}          # fpath → opened file object
        self._lru: list[str] = []                 # ordered by last access
        self._lock = threading.RLock()
        self._max_open = max_open_files
        self._open_fn: Callable[[str], Any] = _default_open
        self._meta_loaders: dict[str, Callable] = {}  # ext → meta-loader

    # ---- singleton ---------------------------------------------------

    @classmethod
    def instance(cls) -> "FileStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Close all files and reset the singleton (mostly for tests)."""
        if cls._instance is not None:
            cls._instance.clear()
            cls._instance = None

    # ---- config ------------------------------------------------------

    @classmethod
    def set_open_fn(cls, fn: Callable[[str], Any]):
        """Override the function used to open files.

        Must return an object whose channels support ``__getitem__``
        and slicing.  Default: ``TdmsFile(path, metadata_only=True)``.
        """
        cls.instance()._open_fn = fn

    @classmethod
    def register_meta_loader(cls, extension: str, fn: Callable):
        """Register a function that extracts metadata without loading data.

        *fn* receives ``fpath: str`` and returns ``list[dict]`` where
        each dict has keys ``name, dt, y_unit, comment, n_samples``.
        """
        cls.instance()._meta_loaders[extension.lower()] = fn

    # ---- public API --------------------------------------------------

    def get(self, fpath: str):
        """Return the cached or newly-opened file object for *fpath*."""
        with self._lock:
            if fpath in self._cache:
                self._lru.remove(fpath)
                self._lru.append(fpath)
                return self._cache[fpath]

        f = self._open_fn(fpath)
        with self._lock:
            self._cache[fpath] = f
            self._lru.append(fpath)
            self._evict()
        return f

    def release(self, fpath: str):
        """Explicitly close and remove *fpath* from the cache."""
        with self._lock:
            if fpath not in self._cache:
                return
            self._close_file(self._cache[fpath])
            del self._cache[fpath]
            self._lru.remove(fpath)

    def clear(self):
        """Close all cached files and empty the cache."""
        with self._lock:
            for f in self._cache.values():
                self._close_file(f)
            self._cache.clear()
            self._lru.clear()

    def get_meta(self, fpath: str) -> list[dict]:
        """Read metadata for *fpath* without loading data arrays.

        Uses a registered meta-loader matching the file extension.
        """
        ext = fpath.rsplit(".", 1)[-1].lower() if "." in fpath else ""
        loader = self._meta_loaders.get(f".{ext}")
        if loader is not None:
            return loader(fpath)
        # Fall back to reading metadata from the cached file object.
        # For TDMS this is effectively the same as load_tdms_meta.
        from dac.modules.timedata.data_loader import load_tdms_meta
        return load_tdms_meta(fpath)

    # ---- internals ---------------------------------------------------

    def _evict(self):
        while len(self._cache) > self._max_open:
            oldest = self._lru.pop(0)
            self._close_file(self._cache[oldest])
            del self._cache[oldest]

    @staticmethod
    def _close_file(f):
        try:
            f.close()
        except Exception:
            pass
