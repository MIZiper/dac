import inspect
from matplotlib.figure import Figure
from dac.core import DataNode, ContextKeyNode, ActionNode

import numpy as np
from time import sleep

class ActionBase(ActionNode): # needs thread
    QUICK_TASKS = []
    # the tasks to assist config input (e.g. browse files instead of filling manually)
    # NOTE: no thread for running the tasks, keep them simple

# TODO: add batch processing: read data, process with predefined parameters, process, save or export, clean and free memory

class ProcessActionBase(ActionBase):
    def __init__(self, context_key: ContextKeyNode, name: str = None, uuid: str = None) -> None:
        super().__init__(context_key, name, uuid)
        self._progress = lambda i, n: self._message(f"Progress: {i}/{n}")
        self._message = print

    def progress(self, i, n):
        self._progress(i, n)
    def message(self, s):
        self._message(f"[{self.name}]>{s}")

class VisualizeActionBase(ActionBase):
    def __init__(self, context_key: ContextKeyNode, name: str = None, uuid: str = None) -> None:
        super().__init__(context_key, name, uuid)
        self._figure: Figure = None
        self._cids = [] # never overwrite in __call__
        self._widgets = [] # to host evented widgets

        # on_callback to change the node params itself

    @property
    def canvas(self):
        return self._figure.canvas
    
    @property
    def figure(self):
        return self._figure
    
    @figure.setter
    def figure(self, fig: Figure):
        # when a figure assigned to visualization, need to clean the previous events and widgets (that have events)
        canvas = fig.canvas
        self._figure = fig
        if hasattr(canvas, "_cids"):
            for cid in canvas._cids:
                canvas.mpl_disconnect(cid)
        if hasattr(canvas, "_widgets"):
            for widget in canvas._widgets:
                widget.disconnect_events()
        canvas._cids = self._cids = []
        canvas._widgets = self._widgets = []
        fig.clf()

    def post_run(self, *args, **kwargs):
        self.canvas.draw_idle()
        # I have a doubt here, this is not called for SAB.
        # But canvas redrawed, widgets redrawed because of thread ends?

PAB = ProcessActionBase
VAB = VisualizeActionBase

class SimpleAction(ActionBase):
    CAPTION = "Simple action"

class SimpleGlobalAction(PAB):
    CAPTION = "Simple global action"

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
    
    def __init__(self, context_key: ContextKeyNode, name: str = None, uuid: str = None) -> None:
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
        
        n = len(self._SEQUENCE)
        rsts = []

        for i, subact_type in enumerate(self._SEQUENCE):
            name = subact_type.__name__

            def embed_progross(j, m):
                self.progress(i*m+j, n*m)
            def embed_message(s):
                self.message(s)

            subact_params = params.get(name)
            self.message(f"Processing [{name}]")

            action = subact_type(self.context_key)
            action.name = name
            if isinstance(action, VAB):
                action._figure = self.figure # avoid figure.setter
            if isinstance(action, PAB):
                action._progress = embed_progross
                action._message = embed_message

            # action.pre_run()
            rst = action(**subact_params)
            if isinstance(rst, DataNode):
                rsts.append(rst)
            elif isinstance(rst, list):
                rsts.extend(rst)
            else:
                ...
            if isinstance(action, VAB):
                self._cids.extend(action._cids)
                self._widgets.extend(action._widgets)
            # action.post_run()

            self.progress(i+1, n)

        # if len(rsts)==1:
        #     return rsts[0] # error, out_name is not defined
        # else:
        #     return rsts

        return rsts

SAB = SequenceActionBase
    
class A1(PAB):
    def __call__(self, sec: int=5, total: int=None):
        if total:
            for i in range(total):
                self.message(f"Starting ... {i+1}/{total}")
                sleep(sec)
                self.progress(i+1, total)
        else:
            self.message(f"Starting unknown.")
            sleep(sec)
class A2(VAB):
    def __call__(self, amp: float):
        ax = self.figure.gca()
        x = np.linspace(0, 4*np.pi, 400)
        ax.plot(x, amp*np.sin(x))
        sleep(amp)
class A1A2(SAB, seq=[A1, A2]):
    CAPTION = "Sequence action example"
class A1A2_2(SAB, seq=[A1A2, A1, A2]):
    CAPTION = "SAB x SAB"
    
class LoopActionBase(PAB):
    # for the case we want to run action multiple times with different parameters
    # original action output streamed/saved to outside
    def __init_subclass__(cls, action_type: type[ActionBase]) -> None:
        return super().__init_subclass__()
LAB = LoopActionBase

class RemoteProcessActionBase(PAB):
    # distribute the calculation (and container) to remote (cloud)
    ...