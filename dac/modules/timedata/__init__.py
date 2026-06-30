"""Defines the `TimeData` class mainly for high sample rate measurement data.
"""

from dac.core.data import DataBase
import numpy as np

class TimeData(DataBase):
    def __init__(self, name: str = None, uuid: str = None, y: np.ndarray=None, dt: float=1, y_unit: str="-", comment: str="") -> None:
        super().__init__(name, uuid)

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
    
    @property
    def t(self):
        return self.x # combine with t0
    
    def to_bins(self, df: float, overlap: float) -> np.ndarray:
        y = self.y
        batch_N = int( 1/df * self.fs )
        stride_N = int( batch_N * (1-overlap) )
        N_batches = (self.length-batch_N) // stride_N + 1
        stride, = y.strides
        assert N_batches > 0

        batches = np.lib.stride_tricks.as_strided(y, shape=(N_batches, batch_N), strides=(stride*stride_N, stride))
        # batches -= batches_mean
        # # the code above will cause problem, it's a `as_strided` mapping
        # # corresponding values are connected

        return batches
    
    def effective_value(self):
        return np.sqrt(np.mean(self.y**2))


class LazyTimeData(TimeData):
    """TimeData that defers loading the ``y`` array until first access.

    Metadata (``dt``, ``y_unit``, ``comment``, ``length``) is available
    immediately.  The actual data array is loaded on first access to
    ``.y`` via the *load_fn* callable injected by the action that created
    this node.

    All internal book-keeping attributes start with ``_`` so they are
    excluded from ``get_construct_config()`` serialization — a saved
    ``LazyTimeData`` degrades to a plain ``TimeData`` (with either the
    loaded array or an empty placeholder).
    """

    def __init__(self, name=None, uuid=None, dt=1, y_unit="-", comment="",
                 n_samples=0, load_fn=None):
        super().__init__(name=name, uuid=uuid,
                         y=np.empty(0, dtype=np.float64),
                         dt=dt, y_unit=y_unit, comment=comment)
        self._n_samples = n_samples
        self._load_fn = load_fn          # () → np.ndarray
        self._loaded = False

    @property
    def y(self):
        if not self._loaded:
            self._load()
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def length(self):
        if self._loaded:
            return len(self._y)
        return self._n_samples

    @property
    def x(self):
        return np.arange(self.length) * self.dt

    def _load(self):
        if self._load_fn is None:
            return
        self._y = self._load_fn()
        self._loaded = True

    def unload(self):
        self._y = np.empty(0, dtype=np.float64)
        self._loaded = False