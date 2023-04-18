from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow

from matplotlib.figure import Figure

class MainWindowBase(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("DAC Base Window")
        self.resize(1024, 768)

        self._thread_pool = QtCore.QThreadPool.globalInstance()

        self.figure: Figure = None

class ProgressBundle:
    ...

class ProgressWidget4Threads:
    ...

class DacWidget(QtCore.QObject):
    ...