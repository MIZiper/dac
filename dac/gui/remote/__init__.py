"""Remote DAC — bridge for connecting desktop GUI to dac_web server.

Uses pywebview (optional dependency) to embed the dac_web /desktop page
for browsing, loading, and saving projects to/from a remote server.
"""

from dac.gui.remote.dialog import DacWebDialog

__all__ = ["DacWebDialog"]
