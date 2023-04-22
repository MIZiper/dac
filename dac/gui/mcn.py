import gc

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT, FigureCanvasQTAgg
from matplotlib.backend_bases import NavigationToolbar2

class MultiCanvasNavigation(NavigationToolbar2QT):
    def __init__(self, parent, canvas=None, coordinates=True):
        self._dummy_canvas = canvas or FigureCanvasQTAgg()
        super().__init__(self._dummy_canvas, parent, coordinates)

    def switch_canvas(self, from_canvas, to_canvas):
        if from_canvas is None:
            from_canvas = self._dummy_canvas
        if to_canvas is None:
            to_canvas = self._dummy_canvas

        mode = self.mode
        for _id_name in ("_id_press", "_id_release", "_id_drag"):
            from_canvas.mpl_disconnect(getattr(self, _id_name))
        NavigationToolbar2.__init__(self, to_canvas)
        self.mode = mode

class CanvasDock(QtWidgets.QDockWidget):
    sig_activated = QtCore.pyqtSignal()
    sig_closed = QtCore.pyqtSignal()
    
    class EmbedCanvas(FigureCanvasQTAgg):
        def __init__(self, dock: "CanvasDock", figure=None):
            super().__init__(figure)
            self._dock = dock
        def mousePressEvent(self, event):
            self._dock.sig_activated.emit()
            return super().mousePressEvent(event)

    def __init__(self, name, parent: QtWidgets.QMainWindow):
        super().__init__(name, parent)
        self.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        parent.addDockWidget(QtCore.Qt.RightDockWidgetArea, self)
        
        self.figure = figure = Figure()
        self.canvas = canvas = CanvasDock.EmbedCanvas(self, figure)
        self.setContentsMargins(0, 0, 0, 0)
        self.setWidget(canvas)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.parent().removeDockWidget(self)
        self.figure.clear()
        gc.collect()
        self.sig_closed.emit()
        return super().closeEvent(event)

    def activate_me(self):
        self.setStyleSheet("background-color: 'orange';")

    def deactivate_me(self):
        p = self.palette()
        p.setColor(self.backgroundRole(), self.parent().palette().color(QtGui.QPalette.Background))
        self.setPalette(p)

class MainWindowWithMCN(QtWidgets.QMainWindow):
    def __init__(self, parent) -> None:
        super().__init__(parent)

        self._mcn = MultiCanvasNavigation(self)
        self._current_canvasdock: CanvasDock = None

        self.setDockOptions(self.AllowNestedDocks)
        tb = self.addToolBar("Plot Navigation")
        tb.addWidget(self._mcn)

    def add_canvasdock(self, name):
        cd = CanvasDock(name, self)
        cd.sig_activated.connect(self._canvasdock_activated)
        cd.sig_closed.connect(self._canvasdock_closed)
        return cd

    def _canvasdock_activated(self):
        cd: CanvasDock = self.sender()
        if self._current_canvasdock is cd:
            return
        if self._current_canvasdock is not None:
            self._current_canvasdock.deactivate_me()
            self._mcn.switch_canvas(self._current_canvasdock.canvas, cd.canvas)
        else:
            self._mcn.switch_canvas(None, cd.canvas)
        cd.activate_me()
        self._current_canvasdock = cd
        
    def _canvasdock_closed(self):
        cd: CanvasDock = self.sender()
        if self._current_canvasdock is cd:
            self._mcn.switch_canvas(cd.canvas, None)
            self._current_canvasdock = None

if __name__=="__main__":
    import sys
    import numpy as np

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindowWithMCN(None)

    for i in range(5):
        cd = win.add_canvasdock(f"Canvas {i}")
        fig = cd.figure
        ax = fig.gca()
        ax.plot(np.random.random(10_000))

    win.show()
    app.exit(app.exec())