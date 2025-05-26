"""Defines base classes for data handling in the DAC framework.

This module provides `DataBase` as a base for data nodes, 
`SimpleDefinition` for context key nodes, and placeholders 
for `ReferenceBase` and `StatData`.
It outlines concepts for data types and definitions within the framework.
"""
from dac.core import DataNode, ContextKeyNode

r"""Concepts and principles

- Workable without container
- Data types are directly use-able as action required
- Member types are direct objects (not the identifier to get target object from container)
- "Definition"s contain just basic elements (int/float/str/dict/list/bool/...)
- Provide hint (<type hint> or [literal hint]) for members
"""

class DataBase(DataNode):
    QUICK_ACTIONS = []
    # the actions to be performed on individual data node
    # equal to `ActionBase(win, fig, ...)(DataBase())`

class SimpleDefinition(ContextKeyNode):
    pass

class ReferenceBase:
    pass

class StatData:
    pass