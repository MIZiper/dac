"""Desktop bridge abstraction layer.

Defines the IPC protocol and abstract base classes for connecting
a PyQt desktop application with a web frontend running in a WebView.

Protocol (JSON Lines over stdin/stdout for pywebview subprocess):

    PyQt → Web:
      {"action":"sendToWeb","type":"<type>","data":{...}}
      {"action":"setWindowTitle","title":"..."}
      {"action":"setWindowSize","width":800,"height":600}
      {"action":"close"}

    Web → PyQt:
      {"action":"message","type":"<type>","data":{...}}
      {"action":"windowClosed"}
      {"action":"error","message":"..."}
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

ACTION_MESSAGE = "message"
ACTION_WINDOW_CLOSED = "windowClosed"
ACTION_ERROR = "error"

ACTION_SEND_TO_WEB = "sendToWeb"
ACTION_SET_TITLE = "setWindowTitle"
ACTION_SET_SIZE = "setWindowSize"
ACTION_SHOW = "showWindow"
ACTION_HIDE = "hideWindow"
ACTION_CLOSE = "close"
ACTION_RELOAD = "reload"

TYPE_LOAD_CONFIG = "loadConfig"
TYPE_RECEIVE_CONFIG = "receiveConfig"


@dataclass
class BridgeMessage:
    action: str
    msg_type: str = ""
    data: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: str | bytes) -> BridgeMessage | None:
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        return cls(
            action=obj.get("action", ""),
            msg_type=obj.get("type", ""),
            data=obj.get("data", {}),
        )

    @classmethod
    def from_dict(cls, obj: dict) -> BridgeMessage | None:
        if not isinstance(obj, dict):
            return None
        return cls(
            action=obj.get("action", ""),
            msg_type=obj.get("type", ""),
            data=obj.get("data", {}),
        )

    def to_dict(self) -> dict:
        d: dict = {"action": self.action, "type": self.msg_type, "data": self.data}
        if not self.msg_type:
            del d["type"]
        if not self.data:
            del d["data"]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class BaseBridge(ABC):

    @abstractmethod
    def start(self, url: str, title: str = "DAC Desktop",
              width: int = 800, height: int = 600):
        """Launch the web view and begin listening for messages."""

    @abstractmethod
    def send_to_web(self, msg_type: str, data: dict):
        """Push a typed message into the web frontend."""

    @abstractmethod
    def set_window_title(self, title: str):
        """Change the web view window title."""

    @abstractmethod
    def set_window_size(self, width: int, height: int):
        """Resize the web view window."""

    @abstractmethod
    def show_window(self):
        """Show / unhide the web view window."""

    @abstractmethod
    def hide_window(self):
        """Hide the web view window."""

    @abstractmethod
    def reload_page(self):
        """Reload the web page."""

    @abstractmethod
    def close(self):
        """Close the web view and release resources."""


OnMessageCallback = Callable[[BridgeMessage], None]
OnClosedCallback = Callable[[], None]


class BridgeFactory:

    @staticmethod
    def create(backend: str, on_message: OnMessageCallback,
               on_closed: OnClosedCallback) -> BaseBridge:
        if backend == "qt":
            from .qt_bridge import QtWebEngineBridge
            return QtWebEngineBridge(on_message, on_closed)
        elif backend == "pywebview":
            from .pywebview_bridge import PyWebViewBridge
            return PyWebViewBridge(on_message, on_closed)
        else:
            raise ValueError(f"Unknown bridge backend: {backend}")

    @staticmethod
    def available_backends() -> list[str]:
        backends: list[str] = []
        try:
            import webview  # noqa: F401
            backends.append("pywebview")
        except ImportError:
            pass
        try:
            from PyQt5.QtWebEngineWidgets import QWebEngineView  # noqa: F401
            backends.append("qt")
        except ImportError:
            pass
        return backends
