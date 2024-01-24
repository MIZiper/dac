import html
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from io import StringIO

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent, QMouseEvent
from PyQt5.QtWidgets import QMainWindow, QWidget

from dac.core.thread import ThreadWorker


class MainWindowBase(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("DAC Base Window")
        self.resize(1024, 768)

        self._thread_pool = QtCore.QThreadPool.globalInstance()
        self._progress_widget = ProgressWidget4Threads(self)

        self._settings = defaultdict(bool)

        self._log_widget: QtWidgets.QPlainTextEdit = None
        self._ipy_widget: QtWidgets.QWidget = None

    def _create_ui(self):
        self._log_widget = QtWidgets.QPlainTextEdit(parent=self) # the log Level selection?
        self._log_widget.appendHtml(f"<b>The log output:</b> @ {datetime.now():%Y-%m-%d} <br/>")
        self._log_widget.setReadOnly(True)
        self._log_widget.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self._log_widget.hide()

    def _create_menu(self):
        menubar = self.menuBar()
        self._dac_menu = tool_menu = menubar.addMenu("&Tool")

        tool_menu.addSeparator()
        tool_menu.addAction("Toggle log output", self.action_toggle_log_widget, shortcut=Qt.CTRL+Qt.Key_L)
        tool_menu.addAction("Toggle IPyConsole", self.action_toggle_ipy_widget, shortcut=Qt.CTRL+Qt.Key_I)
        tool_menu.addSeparator()
        no_thread_action = QtWidgets.QAction("No threading", tool_menu)
        no_thread_action.setCheckable(True)
        no_thread_action.triggered.connect(lambda: self.action_toggle_setting("no_thread"))
        tool_menu.addAction(no_thread_action)

    def _create_status(self):
        status = DacStatusBar(self)
        self.setStatusBar(status)
        status.addPermanentWidget(self._progress_widget)

    def start_thread_worker(self, worker: ThreadWorker):
        worker.signals.message.connect(self.message)
        worker.signals.error.connect(self.excepthook)
        self._progress_widget.add_worker(worker)
        if self._settings["no_thread"]:
            worker.run()
        else:
            self._thread_pool.start(worker)

    def message(self, msg, log=True):
        self.statusBar().showMessage(msg, 3000)
        if log:
            self._log_widget.appendPlainText(f"{datetime.now():%H:%M:%S} - {msg}")

    def _action_resize_log_widget(self):
        h = self.height() - 60
        w = int(self.width() // 2.5)
        self._log_widget.setGeometry(self.width()-20-w, 30, w, h)

    def action_toggle_log_widget(self):
        if self._log_widget.isVisible():
            self._log_widget.hide()
        else:
            self._action_resize_log_widget()
            self._log_widget.show()
            self._log_widget.horizontalScrollBar().setValue(0)
            self._log_widget.raise_()

    def _action_resize_ipy_widget(self):
        if self._ipy_widget is None:
            return
        h = self.height() - 60
        w = int(self.width() // 2.5)
        self._ipy_widget.setGeometry(20, 30, w, h)

    def action_toggle_ipy_widget(self, **kwargs):
        if self._ipy_widget is None:
            try:
                from qtconsole.inprocess import \
                    QtInProcessKernelManager  # inprocess kernel can push variable
                from qtconsole.rich_jupyter_widget import RichJupyterWidget
            except ImportError:
                self.message("IPython console not available", log=False)
                return

            kernel_manager = QtInProcessKernelManager()
            kernel_manager.start_kernel()
            kernel = kernel_manager.kernel

            # if not hasattr(kernel, "io_loop"): # https://github.com/ipython/ipykernel/issues/319
            #     import ipykernel.kernelbase
            #     ipykernel.kernelbase.Kernel.start(kernel)
            # # "io_loop" issue got resolved in new version?

            kernel_client = kernel_manager.client()
            kernel_client.start_channels()

            ipython_widget = RichJupyterWidget(parent=self)
            ipython_widget.kernel_manager = kernel_manager
            ipython_widget.kernel_client = kernel_client

            ipython_widget.exit_requested.connect(lambda: ipython_widget.hide())

            self._ipy_widget = ipython_widget
        else:
            ipython_widget: RichJupyterWidget = self._ipy_widget

        if ipython_widget.isVisible() and not kwargs:
            ipython_widget.hide()
        else:
            kernel = ipython_widget.kernel_manager.kernel
            kernel.shell.push(kwargs)
            self._action_resize_ipy_widget()
            ipython_widget.show()
            ipython_widget.raise_()

    def action_toggle_setting(self, key):
        self._settings[key] = not self._settings[key]

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        if self._log_widget.isVisible():
            self._action_resize_log_widget()
        if self._ipy_widget is not None and self._ipy_widget.isVisible():
            self._action_resize_ipy_widget()
        return super().resizeEvent(a0)

    def excepthook(self, etype, evalue, tracebackobj):
        self._log_widget.appendHtml(f"<br/><b><font color='red'>{etype.__name__}:</font></b> {evalue}")
        info_stream = StringIO()
        traceback.print_tb(tracebackobj, file=info_stream)
        info_stream.seek(0)
        info_str = info_stream.read()
        escaped_str = html.escape(info_str).replace('\n', '<br/>').replace(' ', '&nbsp;')
        self._log_widget.appendHtml(f"<div style='font-family:Consolas'>{escaped_str}</div>")

        self.message("Error occurred, check in log output <Ctrl-L>", log=False)

    def show(self) -> None:
        sys.excepthook = self.excepthook
        return super().show()
    
    def closeEvent(self, a0: QCloseEvent) -> None:
        # TODO: kill the threads in threadpool
        if self._ipy_widget is not None:
            self._ipy_widget.kernel_client.stop_channels()
            self._ipy_widget.kernel_manager.shutdown_kernel()
        return super().closeEvent(a0)

class ProgressBundle(QtWidgets.QWidget):
    def __init__(self, caption):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self._progressbar = progress_bar = QtWidgets.QProgressBar()
        progress_bar.setTextVisible(False)
        progress_bar.setMaximum(0)
        progress_bar.setFixedHeight(6)
        self._caption = caption
        self._label = label = QtWidgets.QLabel("<b style='color:orange;'>(Hold)</b> " + caption)
        layout.addWidget(label)
        layout.addWidget(progress_bar)

    def progress(self, i, n):
        self._progressbar.setMaximum(n)
        self._progressbar.setValue(i)

    def started(self):
        self._label.setText(self._caption)

    # TODO: dbl-click to cancel thread

class ProgressWidget4Threads(QtWidgets.QWidget):
    def __init__(self, parent) -> None:
        super().__init__(parent=parent)
        self._layout = QtWidgets.QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setMinimumHeight(28)

    def add_worker(self, worker: ThreadWorker):
        progress_widget = ProgressBundle(worker.caption)
        worker.signals.progress.connect(progress_widget.progress)
        worker.signals.started.connect(progress_widget.started)
        def finished():
            self._layout.removeWidget(progress_widget)
        worker.signals.finished.connect(finished)
        self._layout.addWidget(progress_widget)
        # the original idea was to automatically switch among progress with one progressbar

class DacStatusBar(QtWidgets.QStatusBar):
    def mouseDoubleClickEvent(self, a0: QMouseEvent) -> None:
        self.parentWidget().action_toggle_log_widget()
        return super().mouseDoubleClickEvent(a0)
    

    
if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindowBase()

    win._create_ui()
    win._create_menu()
    win._create_status()

    win.show()
    app.exit(app.exec())