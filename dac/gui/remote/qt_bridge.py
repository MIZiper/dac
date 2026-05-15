"""QtWebEngineBridge — embeds a QWebEngineView as a PyQt widget.

Uses console.log interception for Web -> Desktop and runJavaScript for
Desktop -> Web communication.

Note: This backend is kept for reference but requires PyQtWebEngine
which is an optional dependency. The primary backend is pywebview.
"""

from __future__ import annotations

from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage

from .bridge import (
    BridgeMessage, BaseBridge, OnMessageCallback, OnClosedCallback,
)

BRIDGE_PREFIX = "DAC_BRIDGE:"


class _BridgePage(QWebEnginePage):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bridge_message_received = None

    def javaScriptConsoleMessage(self, level, msg, line, source):
        if msg.startswith(BRIDGE_PREFIX):
            payload = msg[len(BRIDGE_PREFIX):]
            if self.bridge_message_received:
                self.bridge_message_received(payload)


class QtWebEngineBridge(BaseBridge):

    def __init__(self, on_message: OnMessageCallback,
                 on_closed: OnClosedCallback):
        self._on_message = on_message
        self._on_closed = on_closed

        self._page = _BridgePage()
        self._page.bridge_message_received = self._on_bridge_raw

        self._web_view = QWebEngineView()
        self._web_view.setPage(self._page)
        self._web_view.setMinimumWidth(400)

        self._started = False

    def start(self, url: str, title: str = "DAC Desktop",
              width: int = 800, height: int = 600):
        if self._started:
            return
        self._started = True
        self._web_view.loadFinished.connect(self._on_page_loaded)
        self._web_view.setUrl(QUrl(url))

    def send_to_web(self, msg_type: str, data: dict):
        js = (
            "window.__desktopBridgeReceive("
            + _js_obj({"type": msg_type, **data})
            + ");"
        )
        self._web_view.page().runJavaScript(js)

    def set_window_title(self, title: str):
        pass

    def set_window_size(self, width: int, height: int):
        pass

    def show_window(self):
        pass

    def hide_window(self):
        pass

    def reload_page(self):
        self._web_view.reload()

    def close(self):
        pass

    def get_widget(self) -> QWidget:
        return self._web_view

    def _on_page_loaded(self, ok: bool):
        if not ok:
            self._on_closed()
            return
        self._web_view.page().runJavaScript("""
window.dacDesktop = {
    loadConfig: function(projectId, title, configJson) {
        var msg = JSON.stringify({
            action: "message",
            type: "loadConfig",
            data: {
                projectId: projectId,
                title: title,
                configJson: configJson
            }
        });
        console.log("DAC_BRIDGE:" + msg);
    }
};
window.__desktopBridgeReceive = function(data) {
    if (typeof window.desktopReceiveConfig === 'function' && data.type === 'receiveConfig') {
        window.desktopReceiveConfig(data.title, data.configJson);
    }
};
        """)

    def _on_bridge_raw(self, payload: str):
        msg = BridgeMessage.from_json(payload)
        if msg:
            self._on_message(msg)


def _js_obj(d: dict) -> str:
    import json as _json
    return _json.dumps(d, ensure_ascii=False)
