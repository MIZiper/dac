from dataclasses import dataclass
from . import DataNode, DataClassNode

class DataBase(DataNode):
    QUICK_ACTIONS = []
    # the actions to be performed on individual data node
    # equal to `ActionBase(win, fig, ...)(DataBase())`

@dataclass(eq=False)
class SimpleDefinition(DataClassNode):
    name: str