from . import DataNode

class DataBase(DataNode):
    QUICK_ACTIONS = []
    # the actions to be performed on individual data node
    # equal to `ActionBase(win, fig, ...)(DataBase())`

class SimpleDefinition(DataNode):
    pass