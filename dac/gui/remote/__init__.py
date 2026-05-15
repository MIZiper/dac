"""Remote DAC — bridge for connecting desktop GUI to dac_web server.

Uses pywebview (optional dependency) to open the dac_web /desktop page
for browsing, loading, and saving projects to/from a remote server.

The bridge callbacks are thread-safe: a QObject helper marshals
subprocess stdout callbacks onto the Qt main thread via signals.
"""

from dac.gui.remote.bridge import BridgeFactory, BridgeMessage
from dac.gui.remote.pywebview_bridge import PyWebViewBridge

__all__ = ["BridgeFactory", "BridgeMessage", "PyWebViewBridge"]
