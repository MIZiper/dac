from . import ActionNode

class ActionBase(ActionNode): # needs thread
    ...

class ProcessActionBase(ActionBase):
    ...

class VisualizeActionBase(ActionBase):
    ...

PAB = ProcessActionBase
VAB = VisualizeActionBase