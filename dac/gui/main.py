from PyQt5 import QtWidgets, QtCore

from dac.gui import MainWindowBase

class MainWindow(MainWindowBase):
    def _create_ui(self):
        ...
    
    def _create_menu(self):
        ...

    def _create_status(self):
        ...

    def _route_signals(self):
        ...

if __name__=="__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exit(app.exec())