from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow

from matplotlib.figure import Figure

from dac.core.thread import ThreadWorker

class MainWindowBase(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("DAC Base Window")
        self.resize(1024, 768)

        self._thread_pool = QtCore.QThreadPool.globalInstance()

        self.figure: Figure = None

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

class DacWidget(QtCore.QObject):
    ...