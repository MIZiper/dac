from matplotlib.figure import Figure

from dac.core import DataNode
from . import ActionNode

class ActionBase(ActionNode): # needs thread
    QUICK_TASKS = []
    # the tasks to assist config input (e.g. browse files instead of filling manually)
    # NOTE: no thread for running the tasks, keep them simple

class ProcessActionBase(ActionBase):
    ...

class VisualizeActionBase(ActionBase):
    def __init__(self, context_key: DataNode, name: str = None, uuid: str = None) -> None:
        super().__init__(context_key, name, uuid)
        self._figure: Figure = None
        self._cids = [] # never overwrite in __call__

    @property
    def canvas(self):
        return self._figure.canvas
    
    @property
    def figure(self):
        return self._figure
    
    @figure.setter
    def figure(self, fig: Figure):
        canvas = fig.canvas
        self._figure = fig
        fig.clf()
        if hasattr(canvas, "_cids"):
            for cid in canvas._cids:
                canvas.mpl_disconnect(cid)
        canvas._cids = self._cids = []

    def pre_run(self, *args, **kwargs):
        return super().pre_run(*args, **kwargs)
    
    def post_run(self, *args, **kwargs):
        self.canvas.draw_idle()

PAB = ProcessActionBase
VAB = VisualizeActionBase

class SimpleAction(ActionBase):
    CAPTION = "Simple action"

class SimpleGlobalAction(ActionBase):
    CAPTION = "Simple global action"

class Separator(ActionBase):
    CAPTION = "--- [Separator] ---"
    
class RemoteProcessActionBase(PAB):
    # distribute the calculation (and container) to remote (cloud)
    ...