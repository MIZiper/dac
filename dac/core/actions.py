from . import ActionNode

class ActionBase(ActionNode): # needs thread
    QUICK_TASKS = []
    # the tasks to assist config input (e.g. browse files instead of filling manually)
    # NOTE: no thread for running the tasks, keep them simple

class ProcessActionBase(ActionBase):
    ...

class VisualizeActionBase(ActionBase):
    def pre_run(self, *args, **kwargs):
        # canvas=self.parent_win.figure.canvas
        # for cid in self._cids:
        #     canvas.mpl_disconnect(cid)
        # self._cids.clear()
        return super().pre_run(*args, **kwargs)
    
    def post_run(self, *args, **kwargs):
        # self._cids = action._cids
        return super().post_run(*args, **kwargs)

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