"""Snippet scope

This module provides a module scope for dynamically generated code.

In some scenarios, the analysis can be very temporary and limited to specific data.
It doesn't need to create complete new dac-modules.
So the analysis is also defined inside configuration file, under config['dac']['exec'].
The defined class will reside under this module `dac.core.snippet`.
"""

def exec_script(script: str):
    exec(script, globals=globals())