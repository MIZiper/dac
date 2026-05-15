"""DacWebDialog — remote connection dialog for dac_web server.

Opens a pywebview window showing the dac_web /desktop page, handles
bridge communication, and syncs project config with the main DAC window.
"""

from __future__ import annotations

import json

from PyQt5.QtCore import QSettings, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAction, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

from dac.gui.remote.bridge import BridgeFactory, BridgeMessage


class DacWebDialog(QMainWindow):
    """Window managing the bridge to a dac_web server.

    Signals:
        config_loaded(dict, str): emitted when a project config is received
            from the web. Args: (config_dict, project_id).
    """

    config_loaded = pyqtSignal(dict, str)

    def __init__(self, dac_web_url: str, parent: QMainWindow = None):
        super().__init__(parent)
        self._dac_web_url = dac_web_url.rstrip("/")
        self._project_id: str | None = None
        self._project_title: str = ""
        self._main_win = parent

        self.setWindowTitle(f"DAC Web — {self._dac_web_url}")
        self.resize(500, 200)

        self._backend = self._detect_backend()

        self._create_ui()

        self.bridge = BridgeFactory.create(
            self._backend,
            on_message=self._on_bridge_message,
            on_closed=self._on_bridge_closed,
        )

        desktop_url = f"{self._dac_web_url}/desktop"
        self.bridge.start(desktop_url, title=f"DAC Web — {self._dac_web_url}",
                          width=900, height=650)
        self._status_label.setText(f"Connecting to {self._dac_web_url} ...")

    def _detect_backend(self) -> str:
        available = BridgeFactory.available_backends()
        return available[0] if available else "pywebview"

    def _create_ui(self):
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._status_label = QLabel("Initialising ...")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self._status_label)

        info = QLabel(
            "The DAC Web page opens in a separate pywebview window.\n"
            "Browse projects there and click 'Open in Desktop' to load one here.\n"
            "After editing locally, click 'Push Current Config to Web' below,\n"
            "then click 'Save Back to Server' in the web view to persist."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666;")
        layout.addWidget(info)

        btn_layout = QHBoxLayout()
        self._push_btn = QPushButton("Push Current Config to Web")
        self._push_btn.setMinimumHeight(36)
        self._push_btn.clicked.connect(self._on_push_config)
        self._push_btn.setEnabled(False)
        btn_layout.addWidget(self._push_btn)
        layout.addLayout(btn_layout)

        layout.addStretch()
        self.setCentralWidget(central)

        tool = QToolBar("Web View")
        tool.setMovable(False)
        self._reload_act = QAction("Reload Page", self)
        self._reload_act.triggered.connect(self._reload_page)
        tool.addAction(self._reload_act)
        self._toggle_act = QAction("Show/Hide WebView", self)
        self._toggle_act.setCheckable(True)
        self._toggle_act.setChecked(True)
        self._toggle_act.triggered.connect(self._toggle_webview)
        tool.addAction(self._toggle_act)
        self.addToolBar(Qt.TopToolBarArea, tool)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage(
            f"Web view will open in a separate window. Connected to {self._dac_web_url}"
        )

    def _on_bridge_message(self, msg: BridgeMessage):
        if msg.msg_type == "loadConfig":
            data = msg.data
            config_json = data.get("configJson", "{}")
            self._project_id = data.get("projectId", "")
            self._project_title = data.get("title", "")
            try:
                config = json.loads(config_json)
            except json.JSONDecodeError:
                self._status_label.setText(
                    f"<b style='color:red;'>Error: Invalid config JSON</b>"
                )
                self._status_bar.showMessage("Received invalid config JSON from web")
                return
            self._status_label.setText(
                f"<b>Loaded:</b> {self._project_title}<br/>"
                f"<span style='color:#666;'>Project ID: {self._project_id}</span>"
            )
            self._push_btn.setEnabled(True)
            self._status_bar.showMessage(f"Loaded project: {self._project_title}")
            self.config_loaded.emit(config, self._project_id)
        elif msg.msg_type == "bridgeError":
            err = msg.data.get("message", "Unknown bridge error")
            self._status_label.setText(
                f"<b style='color:red;'>Bridge Error:</b><br/>{err}"
            )
            self._status_bar.showMessage(f"Bridge error: {err}")

    def _on_bridge_closed(self):
        self._status_label.setText(
            "<b style='color:orange;'>Web view closed.</b><br/>"
            "Reopen the connection to continue."
        )
        self._status_bar.showMessage("Web view disconnected.")

    def _on_push_config(self):
        if not self._project_id:
            self._status_bar.showMessage("No project loaded from web")
            return
        if not self._main_win or self._main_win.container is None:
            self._status_bar.showMessage("No project open in main window")
            return
        try:
            config = self._main_win.get_config()
        except Exception as e:
            self._status_bar.showMessage(f"Failed to get config: {e}")
            return
        dac_config = config.get("dac", config)
        config_json = json.dumps(dac_config, ensure_ascii=False)
        title = self._main_win.windowTitle().split(" | ")[0]
        self.bridge.send_to_web("receiveConfig", {
            "title": title,
            "configJson": config_json,
        })
        self._status_bar.showMessage(
            f"Config pushed to web — click 'Save Back to Server' in the web view"
        )
        self.bridge.show_window()

    def _reload_page(self):
        if hasattr(self, "bridge") and self.bridge:
            self.bridge.reload_page()
            self._status_bar.showMessage("Web page reloading...")

    def _toggle_webview(self, checked: bool):
        if checked:
            self.bridge.show_window()
        else:
            self.bridge.hide_window()

    def send_current_config(self, title: str, config: dict):
        dac_config = config.get("dac", config)
        config_json = json.dumps(dac_config, ensure_ascii=False)
        self.bridge.send_to_web("receiveConfig", {
            "title": title,
            "configJson": config_json,
        })

    def closeEvent(self, event):
        if hasattr(self, "bridge") and self.bridge:
            self.bridge.close()
        super().closeEvent(event)
