"""Snippet scope

This module provides a module scope for dynamically generated code.

In some scenarios, the analysis can be very temporary and limited to specific data.
It doesn't need to create complete new dac-modules.
So the analysis is also defined inside configuration file, under config['dac']['exec'].
The defined class will reside under this module `dac.core.snippet`.
"""

from dac.core.exceptions import SnippetError


def exec_script(script: str):
    if not script:
        return
    try:
        exec(script, globals=globals())
    except Exception as e:
        raise SnippetError(f"Failed to execute snippet: {e}") from e