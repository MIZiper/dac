from PyQt5 import QtWidgets, QtCore

from dac.gui import MainWindowBase

class MainWindow(MainWindowBase):
    ...

if __name__=="__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exit(app.exec())