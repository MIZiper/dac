from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QMainWindow, QWidget, QTreeWidget, QTreeWidgetItem, QStyle

from matplotlib.figure import Figure

from dac.core import Container, ActionNode, DataNode, GCK
from dac.core.thread import ThreadWorker

NAME, TYPE, REMARK = range(3)

class MainWindowBase(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("DAC Base Window")
        self.resize(1024, 768)

        self._thread_pool = QtCore.QThreadPool.globalInstance()
        self._progress_widget = ProgressWidget4Threads(self)

        self.figure: Figure = None

    def _create_ui(self):
        ...

    def _create_menu(self):
        ...

    def _create_status(self):
        status = self.statusBar()
        status.addPermanentWidget(self._progress_widget)

    def start_thread_worker(self, worker: ThreadWorker):
        # TODO: worker.signals.message.connect()
        self._progress_widget.add_worker(worker)
        self._thread_pool.start(worker)

    def message(self, msg):
        self.statusBar().showMessage(msg, 3000)

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

class DataListWidget(QTreeWidget):
    sig_edit_data_requested = QtCore.pyqtSignal(DataNode)
    
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._STYLE = self.style()

        self._container: Container = None

        self.setHeaderLabels(["Name", "Type", "Remark"])
        self.setColumnWidth(NAME, 150)
        self.setColumnWidth(TYPE, 200)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.action_context_requested)
        self.itemClicked.connect(self.action_item_clicked)

    def refresh(self, container: Container=None):
        self.clear()
        if container is None:
            container = self._container
            if container is None:
                return
        else:
            self._container = container

        global_item = QtWidgets.QTreeWidgetItem(self)
        global_item.setText(NAME, "N/A")
        global_item.setText(TYPE, "Global Nodes")
        global_item.setData(NAME, Qt.ItemDataRole.UserRole, GCK)
        for node_type, node_name, node_object in container.GlobalContext.NodeIter:
            itm = QtWidgets.QTreeWidgetItem(global_item)
            itm.setText(NAME, node_name)
            itm.setText(TYPE, node_type.__name__)
            itm.setData(NAME, Qt.ItemDataRole.UserRole, node_object)
            if container._current_key is node_object:
                itm.setIcon(NAME, self._STYLE.standardIcon(QStyle.StandardPixmap.SP_CommandLink))
        global_item.setExpanded(True)

        local_item = QtWidgets.QTreeWidgetItem(self)
        local_item.setText(NAME, "N/A")
        local_item.setText(TYPE, "Local Nodes")
        if container._current_key is not GCK:
            local_item.setText(NAME, container._current_key.name)
            for node_type, node_name, node_object in container.CurrentContext.NodeIter:
                itm = QtWidgets.QTreeWidgetItem(local_item)
                itm.setText(NAME, node_name)
                itm.setText(TYPE, node_type.__name__)
                itm.setDisabled(True)
        local_item.setExpanded(True)

    def action_context_requested(self, pos: QtCore.QPoint):
        if (container := self._container) is None:
            return
        itm = self.itemAt(pos)
        menu = QtWidgets.QMenu("DataMenu")

        if (not itm) or not (node_object := itm.data(NAME, Qt.ItemDataRole.UserRole)):
            return
        
        if node_object is GCK:
            def cb_creation_gen(n_t: type[DataNode]):
                def cb_creation():
                    new_node = n_t(name="[New node]")
                    container.GlobalContext.add_node(new_node)
                    self.refresh()
                return cb_creation
            
            for n_t in Container.GetGlobalDataTypes():
                menu.addAction(n_t.__name__).triggered.connect(cb_creation_gen(n_t))
        else:
            def cb_activate_gen(key_object):
                def cb_activate():
                    container.activate_context(key_object)
                    # TODO: notify action_list
                    self.refresh()
                return cb_activate
            
            if node_object is container._current_key:
                menu.addAction("De-activate").triggered.connect(cb_activate_gen(GCK))
            else:
                menu.addAction("Activate").triggered.connect(cb_activate_gen(node_object))

            def cb_del_gen(key_object):
                def cb_del():
                    if key_object is container._current_key:
                        container.activate_context(GCK)
                        # TODO: notify action_list

                    container.remove_global_node(key_object)
                    self.refresh()

                return cb_del
            
            menu.addAction("Delete").triggered.connect(cb_del_gen(node_object))

        menu.exec(self.viewport().mapToGlobal(pos))

    def action_item_clicked(self, item: QTreeWidgetItem, col: int):
        data = item.data(NAME, Qt.ItemDataRole.UserRole)
        self.sig_edit_data_requested.emit(data)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        # mid-btn-click => copy name. mid-button-click won't trigger 'itemClicked'
        if e.button()==Qt.MouseButton.MidButton:
            itm = self.itemAt(e.pos())
            name = itm.text(NAME)
            QtWidgets.QApplication.clipboard().setText(name)
        return super().mousePressEvent(e)

class ActionListWidget(QTreeWidget):
    sig_edit_action_requested = QtCore.pyqtSignal(ActionNode)
    
    PIXMAP = {
        ActionNode.ActionStatus.INIT: QStyle.StandardPixmap.SP_FileIcon,
        ActionNode.ActionStatus.CONFIGURED: QStyle.StandardPixmap.SP_FileDialogContentsView,
        ActionNode.ActionStatus.COMPLETE: QStyle.StandardPixmap.SP_DialogApplyButton,
        ActionNode.ActionStatus.FAILED: QStyle.StandardPixmap.SP_DialogCancelButton,
    }

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._STYLE = self.style()

        self._container: Container = None
        self._cids = []

        self.setHeaderLabels(["Name", "Output", "Remark"])
        self.setColumnWidth(NAME, 200)
        self.setColumnWidth(TYPE, 150)
        self.setSelectionMode(self.SelectionMode.ExtendedSelection)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.action_context_requested)
        self.itemClicked.connect(self.action_item_clicked)
        self.itemDoubleClicked.connect(self.action_item_dblclicked)

    def refresh(self, container: Container=None):
        self.clear()
        if container is None:
            container = self._container
            if container is None:
                return
        else:
            self._container = container

        for action in container.ActionsInCurrentContext:
            itm = QtWidgets.QTreeWidgetItem(self)
            itm.setText(NAME, action.name)
            itm.setData(NAME, Qt.ItemDataRole.UserRole, action)
            if action.out_name is not None:
                itm.setText(TYPE, action.out_name)

            itm.setIcon(NAME, self._STYLE.standardIcon(ActionListWidget.PIXMAP[action.status]))

    def run_action(self, action: ActionNode):
        if (container := self._container) is None:
            return
        params = container.prepare_params_for_action(action)

        

        action.status = ActionNode.ActionStatus.COMPLETE # TODO: update accordingly
        self.refresh()

    def action_context_requested(self, pos: QtCore.QPoint):
        if (container := self._container) is None:
            return
        itms = self.selectedItems()
        menu = QtWidgets.QMenu("ActionMenu")

        if not itms:
            def cb_creation_gen(a_t: type[ActionNode]):
                def cb_creation():
                    a = a_t(container._current_key)
                    container.actions.append(a)
                    self.refresh()
                return cb_creation
            
            for a_t in container.ActionTypesInCurrentContext:
                if a_t is None:
                    menu.addSeparator()
                else:
                    menu.addAction(a_t.CAPTION).triggered.connect(cb_creation_gen(a_t))
        else:
            acts = [itm.data(NAME, Qt.ItemDataRole.UserRole) for itm in itms]

            def cb_del_gen(aa: list[ActionNode]):
                def cb_del():
                    for a in aa:
                        container.actions.remove(a)
                    self.refresh()
                    
                return cb_del
            
            menu.addAction("Delete").triggered.connect(cb_del_gen(acts))

        menu.exec(self.viewport().mapToGlobal(pos))
    
    def action_item_clicked(self, item: QTreeWidgetItem, col: int):
        act = item.data(NAME, Qt.ItemDataRole.UserRole)
        self.sig_edit_action_requested.emit(act)

    def action_item_dblclicked(self, item: QTreeWidgetItem, col: int):
        self.run_action( item.data(NAME, Qt.ItemDataRole.UserRole) )
        

if __name__=="__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindowBase()
    win.show()
    app.exit(app.exec())