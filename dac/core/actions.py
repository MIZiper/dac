from . import ActionNode

class ActionBase(ActionNode): # needs thread
    QUICK_TASKS = []
    # the tasks to assist config input (e.g. browse files instead of filling manually)
    # NOTE: no thread for running the tasks, keep them simple

class ProcessActionBase(ActionBase):
    ...

class VisualizeActionBase(ActionBase):
    ...

PAB = ProcessActionBase
VAB = VisualizeActionBase

class SimpleAction(ActionBase):
    CAPTION = "Simple Action"

class SimpleGlobalAction(ActionBase):
    CAPTION = "Simple Global Action"

class RemoteProcessActionBase(PAB):
    # distribute the calculation (and container) to remote (cloud)
    ...