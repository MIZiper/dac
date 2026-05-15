"""PyWebViewBridge — manages pywebview as a subprocess.

Communicates with the subprocess via stdin/stdout using the JSON Lines protocol.
The subprocess entry point is pywebview_host.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

from .bridge import (
    BridgeMessage, BaseBridge, OnMessageCallback, OnClosedCallback,
    ACTION_SEND_TO_WEB, ACTION_SET_TITLE, ACTION_SET_SIZE,
    ACTION_SHOW, ACTION_HIDE, ACTION_CLOSE,
    ACTION_ERROR, ACTION_RELOAD,
)


class PyWebViewBridge(BaseBridge):

    def __init__(self, on_message: OnMessageCallback,
                 on_closed: OnClosedCallback):
        self._on_message = on_message
        self._on_closed = on_closed
        self._process: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._running = False
        self._host_script: str | None = None
        self._last_error = ""

    def start(self, url: str, title: str = "DAC Desktop",
              width: int = 800, height: int = 600):
        if self._running:
            return

        self._host_script = self._find_host_script()

        if not self._check_pywebview():
            self._last_error = (
                "pywebview is not installed. Install it with:\n"
                "    pip install pywebview\n"
                "(requires a GUI backend: GTK on Linux, Cocoa on macOS, Win32 on Windows)"
            )
            self._emit_error()
            return

        cmd = [
            sys.executable,
            self._host_script,
            "--url", url,
            "--title", title,
            "--width", str(width),
            "--height", str(height),
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as e:
            self._last_error = f"Failed to launch pywebview subprocess: {e}"
            self._emit_error()
            return

        self._running = True

        self._reader_thread = threading.Thread(
            target=self._read_stdout, daemon=True
        )
        self._reader_thread.start()

        self._stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self._stderr_thread.start()

        threading.Thread(target=self._monitor_process, daemon=True).start()

    def send_to_web(self, msg_type: str, data: dict):
        self._write_stdin({
            "action": ACTION_SEND_TO_WEB,
            "type": msg_type,
            "data": data,
        })

    def set_window_title(self, title: str):
        self._write_stdin({"action": ACTION_SET_TITLE, "title": title})

    def set_window_size(self, width: int, height: int):
        self._write_stdin({
            "action": ACTION_SET_SIZE,
            "width": width,
            "height": height,
        })

    def show_window(self):
        self._write_stdin({"action": ACTION_SHOW})

    def hide_window(self):
        self._write_stdin({"action": ACTION_HIDE})

    def reload_page(self):
        self._write_stdin({"action": ACTION_RELOAD})

    def close(self):
        if self._running:
            self._write_stdin({"action": ACTION_CLOSE})
        self._cleanup()

    @property
    def last_error(self) -> str:
        return self._last_error

    def _check_pywebview(self) -> bool:
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import webview"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _write_stdin(self, obj: dict):
        if not self._process or not self._process.stdin or not self._running:
            return
        try:
            line = json.dumps(obj, ensure_ascii=False) + "\n"
            self._process.stdin.write(line)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            self._running = False
            self._on_closed()

    def _read_stdout(self):
        if not self._process or not self._process.stdout:
            return
        try:
            for line in self._process.stdout:
                line = line.strip()
                if not line:
                    continue
                msg = BridgeMessage.from_json(line)
                if msg is None:
                    continue
                if msg.action == "windowClosed":
                    self._running = False
                    self._on_closed()
                    break
                elif msg.action == "message":
                    self._on_message(msg)
                elif msg.action == "error":
                    self._last_error = msg.data.get("message", str(msg.data))
        except (OSError, ValueError):
            pass
        finally:
            self._running = False
            self._on_closed()

    def _read_stderr(self):
        if not self._process or not self._process.stderr:
            return
        try:
            for line in self._process.stderr:
                self._last_error += line
        except (OSError, ValueError):
            pass

    def _monitor_process(self):
        if not self._process:
            return
        try:
            self._process.wait()
        except Exception:
            pass
        if self._running:
            self._running = False
            rc = self._process.returncode if self._process else -1
            if rc != 0:
                self._emit_error()
            self._on_closed()

    def _emit_error(self):
        self._on_message(BridgeMessage(
            action=ACTION_ERROR,
            msg_type="bridgeError",
            data={"message": self._last_error or "pywebview subprocess failed"},
        ))

    def _cleanup(self):
        self._running = False
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    @staticmethod
    def _find_host_script() -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, "pywebview_host.py")
