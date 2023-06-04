from matplotlib.figure import Figure
import inspect
from dac.core import DataNode
from . import ActionNode
from .data import SimpleDefinition

from time import sleep

class ActionBase(ActionNode): # needs thread
    QUICK_TASKS = []
    # the tasks to assist config input (e.g. browse files instead of filling manually)
    # NOTE: no thread for running the tasks, keep them simple

class ProcessActionBase(ActionBase):
    def __init__(self, context_key: DataNode, name: str = None, uuid: str = None) -> None:
        super().__init__(context_key, name, uuid)
        self._progress = print
        self._message = print

    def progress(self, i, n):
        self._progress(i, n)
    def message(self, s):
        self._message(s)

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

class SimpleGlobalAction(PAB):
    CAPTION = "Simple global action"
    def __call__(self, sim: SimpleDefinition, sec: int=5, total: int=None):
        if total:
            for i in range(total):
                self.message(f"Starting ... {i+1}/{total}")
                sleep(sec)
                self.progress(i+1, total)
        else:
            self.message(f"Starting unknown.")
            sleep(sec)

class Separator(ActionBase):
    CAPTION = "--- [Separator] ---"
    def __call__(self):
        pass

class SequenceActionBase(PAB, VAB):
    CAPTION = "Not implemented sequence"
    _SEQUENCE = []

    def __init_subclass__(cls, seq: list[type[ActionNode]]) -> None:
        cls._SEQUENCE = seq
        
        signatures = {}
        for act_type in seq:
            signatures[act_type.__name__] = act_type._SIGNATURE

        cls._SIGNATURE = signatures
    
    def __init__(self, context_key: DataNode, name: str = None, uuid: str = None) -> None:
        super().__init__(context_key, name, uuid)
        self._construct_config = SequenceActionBase._GetCCFromS(self._SIGNATURE)

    @staticmethod
    def _GetCCFromS(sig: inspect.Signature | dict):
        # get construct config from signature
        cfg = {}
        
        if isinstance(sig, inspect.Signature):
            for key, param in sig.parameters.items():
                if key=="self":
                    continue
                elif param.default is not inspect._empty:
                    cfg[key] = param.default
                elif param.annotation is not inspect._empty:
                    cfg[key] = ActionNode.Annotation2Config(param.annotation)
                else:
                    cfg[key] = "<Any>"
            if (ret_ann:=sig.return_annotation) is not inspect._empty and ret_ann.__name__!="list":
                ... # how to pass the return result?
        else:
            for subact_name, subact_sig in sig.items():
                cfg[subact_name] = SequenceActionBase._GetCCFromS(subact_sig)

        return cfg
    
    def __call__(self, **params):
        # using dict => each action type can be used only once

        def embed_progross():
            ...
        
        n = len(self._SEQUENCE)
        for i, subact_type in enumerate(self._SEQUENCE):
            subact_params = params.get(subact_type.__name__)
            self.message()
            if issubclass(subact_type, VAB):
                ...
            elif issubclass(subact_type, PAB):
                ...
            self.progress(i+1, n)

SAB = SequenceActionBase
    
class A1(PAB):
    pass
class A2(VAB):
    pass
class A1A2(SAB, seq=[SimpleGlobalAction, A2]):
    pass
class A1A2Simple(SAB, seq=[A1A2, SimpleGlobalAction]):
    CAPTION = "Multiple actions"
    
class RemoteProcessActionBase(PAB):
    # distribute the calculation (and container) to remote (cloud)
    ...