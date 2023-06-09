from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QMainWindow, QWidget, QTreeWidget, QTreeWidgetItem, QStyle
from PyQt5.Qsci import QsciScintilla, QsciLexerYAML

import sys
from matplotlib.figure import Figure
import yaml
import html
import traceback
from io import StringIO
from functools import partial
from datetime import datetime

from dac.core import Container, ActionNode, DataNode, GCK, NodeBase
from dac.core.actions import ActionBase, VAB, PAB
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
        self._log_widget = QtWidgets.QPlainTextEdit(parent=self) # the log Level selection?
        self._log_widget.appendHtml(f"<b>The log output:</b> @ {datetime.now():%Y-%m-%d} <br/>")
        self._log_widget.setReadOnly(True)
        self._log_widget.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self._log_widget.hide()

    def _create_menu(self):
        menubar = self.menuBar()
        self._dac_menu = tool_menu = menubar.addMenu("&Tool")

        tool_menu.addAction("Toggle log output", self.action_toggle_log_widget, shortcut=Qt.CTRL+Qt.Key_L)

    def _create_status(self):
        status = self.statusBar()
        status.addPermanentWidget(self._progress_widget)

    def start_thread_worker(self, worker: ThreadWorker):
        worker.signals.message.connect(self.message)
        self._progress_widget.add_worker(worker)
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

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        if self._log_widget.isVisible():
            self._action_resize_log_widget()
        return super().resizeEvent(a0)

    def show(self) -> None:
        def excepthook(etype, evalue, tracebackobj):
            self._log_widget.appendHtml(f"<br/><b><font color='red'>{etype.__name__}:</font></b> {evalue}")
            info_stream = StringIO()
            traceback.print_tb(tracebackobj, file=info_stream)
            info_stream.seek(0)
            info_str = info_stream.read()
            escaped_str = html.escape(info_str).replace('\n', '<br/>').replace(' ', '&nbsp;')
            self._log_widget.appendHtml(f"<div style='font-family:Consolas'>{escaped_str}</div>")

            self.message("Error occurred, check in log output <Ctrl-L>", log=False)
        
        sys.excepthook = excepthook
        return super().show()

class TaskBase:
    def __init__(self, dac_win: MainWindowBase, name: str, *args):
        self.dac_win = dac_win
        self.name = name

    def request_update_action(self):
        pass

    def __call__(self, action: ActionBase):
        pass

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
    sig_action_update_requested = QtCore.pyqtSignal()
    
    def __init__(self, parent: MainWindowBase) -> None:
        super().__init__(parent)
        self._STYLE = self.style()
        self._parent_win = parent
        self._container: Container = None

        self.setHeaderLabels(["Name", "Type", "Remark"])
        self.setColumnWidth(NAME, 150)
        self.setColumnWidth(TYPE, 200)

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

        global_item = QtWidgets.QTreeWidgetItem(self)
        global_item.setText(NAME, "N/A")
        global_item.setText(TYPE, "Global Nodes")
        global_item.setData(NAME, Qt.ItemDataRole.UserRole, GCK)
        for node_type, node_name, node_object in container.GlobalContext.NodeIter:
            itm = QtWidgets.QTreeWidgetItem(global_item)
            itm.setText(NAME, node_name)
            itm.setText(TYPE, node_type.__name__)
            itm.setData(NAME, Qt.ItemDataRole.UserRole, node_object)
            if container.current_key is node_object:
                itm.setIcon(NAME, self._STYLE.standardIcon(QStyle.StandardPixmap.SP_CommandLink))
        global_item.setExpanded(True)

        local_item = QtWidgets.QTreeWidgetItem(self)
        local_item.setText(NAME, "N/A")
        local_item.setText(TYPE, "Local Nodes")
        if container.current_key is not GCK:
            local_item.setText(NAME, container.current_key.name)
            for node_type, node_name, node_object in container.CurrentContext.NodeIter:
                itm = QtWidgets.QTreeWidgetItem(local_item)
                itm.setText(NAME, node_name)
                itm.setText(TYPE, node_type.__name__)
                itm.setData(NAME, Qt.ItemDataRole.UserRole, node_object)
                itm.setData(TYPE, Qt.ItemDataRole.UserRole, True) # mark as un-editable
                # itm.setDisabled(True) # TODO: enable for editing, and quick_actions context menu
        local_item.setExpanded(True)

    def action_context_requested(self, pos: QtCore.QPoint):
        if (container := self._container) is None:
            return
        itm = self.itemAt(pos)
        menu = QtWidgets.QMenu("DataMenu")

        if (not itm) or not (node_object := itm.data(NAME, Qt.ItemDataRole.UserRole)):
            return
        
        if getattr(node_object, "QUICK_ACTIONS", []):
            nodes = []
            for i in self.selectedItems():
                node = i.data(NAME, Qt.ItemDataRole.UserRole)
                if type(node) is type(node_object): # or subclass?
                    nodes.append(node)

            def cb_quickaction_gen(qat: tuple[type[ActionBase], str, dict], data_nodes: list[DataNode]):
                act_type, data_param_name, other_params = qat
                def cb_quickaction():
                    params = {data_param_name: data_nodes, **other_params}
                    act = act_type(context_key=container.current_key)
                    act.container = container
                    if isinstance(act, VAB):
                        act.figure = self._parent_win.figure
                    act.pre_run()
                    act(**params)
                    act.post_run()
                return cb_quickaction
            
            for qat  in node_object.QUICK_ACTIONS:
                qat: tuple[type[ActionBase], str, dict]
                act_type, data_param_name, other_params = qat
                menu.addAction(act_type.CAPTION).triggered.connect(cb_quickaction_gen(qat, nodes))
            menu.addSeparator()
        
        if (uneditable:=itm.data(TYPE, Qt.ItemDataRole.UserRole)): # data nodes in local context
            if len(menu.actions()):
                menu.exec(self.viewport().mapToGlobal(pos))
            return # stop here, no activate / delete
        
        if node_object is GCK:
            def cb_creation_gen(n_t: type[DataNode]):
                def cb_creation():
                    new_node = n_t(name="[New node]")
                    container.GlobalContext.add_node(new_node)
                    self.refresh()
                    self.sig_edit_data_requested.emit(new_node)
                return cb_creation
            
            for n_t in Container.GetGlobalDataTypes():
                if isinstance(n_t, str):
                    menu.addAction(n_t).setEnabled(False)
                else:
                    menu.addAction(n_t.__name__).triggered.connect(cb_creation_gen(n_t))
        else:
            def cb_activate_gen(key_object):
                def cb_activate():
                    container.activate_context(key_object)
                    self.sig_action_update_requested.emit()
                    self.refresh()
                return cb_activate
            
            if node_object is container.current_key:
                menu.addAction("De-activate").triggered.connect(cb_activate_gen(GCK))
            else:
                menu.addAction("Activate").triggered.connect(cb_activate_gen(node_object))

            # TODO: run actions of this context in sequence

            def cb_del_gen(key_object):
                def cb_del():
                    if key_object is container.current_key:
                        container.activate_context(GCK)
                        self.sig_action_update_requested.emit()

                    container.remove_global_node(key_object)
                    self.refresh()

                return cb_del
            
            menu.addSeparator()
            menu.addAction("Delete").triggered.connect(cb_del_gen(node_object))

        menu.exec(self.viewport().mapToGlobal(pos))

    def action_item_clicked(self, item: QTreeWidgetItem, col: int):
        data = item.data(NAME, Qt.ItemDataRole.UserRole)
        uneditable = False # item.data(TYPE, Qt.ItemDataRole.UserRole)

        if uneditable or not isinstance(data, DataNode) or data is GCK: # GCK not edit-able
            return
        self.sig_edit_data_requested.emit(data)

    def action_item_dblclicked(self, item: QTreeWidgetItem, col: int):
        uneditable = item.data(TYPE, Qt.ItemDataRole.UserRole)

        if uneditable or (container := self._container) is None or not (node_object := item.data(NAME, Qt.ItemDataRole.UserRole)):
            return
        
        def cb_activate(key_object):
            container.activate_context(key_object)
            self.sig_action_update_requested.emit()
            self.refresh()

        if node_object is container.current_key:
            cb_activate(GCK)
        else:
            cb_activate(node_object)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        # mid-btn-click => copy name. mid-button-click won't trigger 'itemClicked'
        if e.button()==Qt.MouseButton.MidButton:
            itm = self.itemAt(e.pos())
            name = itm.text(NAME)
            QtWidgets.QApplication.clipboard().setText(name)
        return super().mousePressEvent(e)
    
    def action_apply_node_config(self, node: DataNode, config: dict, fire: bool=False):
        if not isinstance(node, DataNode):
            return
        
        # if node from previous container, seems works too
        if (new_name:=config.get("name")) and new_name!=node.name:
            self._container.GlobalContext.rename_node_to(node, new_name)
        node.apply_construct_config(config)
        self.refresh()

class ActionListWidget(QTreeWidget):
    sig_edit_action_requested = QtCore.pyqtSignal(ActionNode)
    sig_data_update_requested = QtCore.pyqtSignal()
    
    PIXMAP = {
        ActionNode.ActionStatus.INIT: QStyle.StandardPixmap.SP_FileIcon,
        ActionNode.ActionStatus.CONFIGURED: QStyle.StandardPixmap.SP_FileDialogContentsView,
        ActionNode.ActionStatus.COMPLETE: QStyle.StandardPixmap.SP_DialogApplyButton,
        ActionNode.ActionStatus.FAILED: QStyle.StandardPixmap.SP_DialogCancelButton,
    }

    def __init__(self, parent: MainWindowBase) -> None:
        super().__init__(parent)
        self._STYLE = self.style()
        self._parent_win = parent
        self._container: Container = None

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
        params = container.prepare_params_for_action(action._SIGNATURE, action._construct_config)

        def completed(rst):
            current_context = container.CurrentContext
            if isinstance(rst, DataNode):
                rst.name = action.out_name # what if out_name is None?
                current_context.add_node(rst)
                self.sig_data_update_requested.emit()
            elif isinstance(rst, list):
                for e_rst in rst:
                    e_rst: DataNode # cautious if e_rst is not DataNode
                    current_context.add_node(e_rst)
                self.sig_data_update_requested.emit()
            else:
                pass # no output or other type_of_data

            action.status = ActionNode.ActionStatus.COMPLETE # TODO: update accordingly
            self.refresh()

        action.container = container
        self._parent_win.message(f"[{action.name}]")

        if isinstance(action, VAB):
            action.figure = self._parent_win.figure

        if isinstance(action, PAB):
            def fn(p, progress_emitter, logger):
                action._progress = progress_emitter
                action._message = logger
                action.pre_run()
                rst = action(**p)
                action.post_run()
                return rst
            worker = ThreadWorker(fn=fn, caption=action.name, p=params)
            worker.signals.result.connect(completed)
            self._parent_win.start_thread_worker(worker)
        else:
            action.pre_run()
            rst = action(**params)
            action.post_run()
            completed(rst)

    def action_context_requested(self, pos: QtCore.QPoint):
        if (container := self._container) is None:
            return
        itms = self.selectedItems()
        menu = QtWidgets.QMenu("ActionMenu")
        menu_stack = []

        if not itms:
            # NOTE: when tree is full (and with scrollbar), it's not easy to trigger
            def cb_creation_gen(a_t: type[ActionNode]):
                def cb_creation():
                    a = a_t(context_key=container.current_key)
                    container.actions.append(a)
                    self.refresh()
                    self.sig_edit_action_requested.emit(a)
                return cb_creation
            
            for a_t in container.ActionTypesInCurrentContext:
                if isinstance(a_t, str):
                    if a_t.endswith(">]"):
                        menu_stack.append(menu)
                        menu = menu.addMenu(a_t)
                    elif a_t.endswith("<]"):
                        menu = menu_stack.pop()
                    else:
                        menu.addAction(a_t).setEnabled(False)
                else:
                    menu.addAction(a_t.CAPTION).triggered.connect(cb_creation_gen(a_t))
        else:
            acts = [itm.data(NAME, Qt.ItemDataRole.UserRole) for itm in itms]

            if len(acts)==1:
                def cb_task_gen(task, act):
                    def request_update_action():
                        self.sig_edit_action_requested.emit(act)

                    def cb_task():
                        task.request_update_action = request_update_action
                        task(act)
                        request_update_action()
                    return cb_task
                act: ActionBase = acts[0]
                for task in act.QUICK_TASKS:
                    task: TaskBase
                    menu.addAction(task.name).triggered.connect(cb_task_gen(task, act))
                menu.addSeparator()

            def cb_cp2_gen(aa: list[ActionNode], context_key: DataNode):
                def cb_cp2():
                    for oa in aa:
                        oac = oa.get_construct_config()

                        a_t = oa.__class__
                        a = a_t(context_key=context_key)

                        a.apply_construct_config(oac)

                        container.actions.append(a)

                    if context_key is container.current_key:
                        self.refresh() # if not copy to self, no need to refresh

                return cb_cp2
            
            def cb_mvaft_gen(aa: list[ActionNode], a: ActionNode):
                def cb_mvaft():
                    for oa in aa:
                        container.actions.remove(oa)
                    idx = container.actions.index(a)
                    
                    container.actions[idx+1:idx+1] = aa

                    self.refresh()

                return cb_mvaft

            def cb_del_gen(aa: list[ActionNode]):
                def cb_del():
                    for a in aa:
                        container.actions.remove(a)
                    self.refresh()
                    
                return cb_del
            
            if container.current_key is not GCK:
                cp2menu = menu.addMenu("Copy to")
                current_type = type(container.current_key)
                for node_type, node_name, node in container.GlobalContext.NodeIter:
                    if isinstance(node, current_type):
                        # only allow copying to context of same type
                        cp2menu.addAction(node_name).triggered.connect(
                            cb_cp2_gen(acts, node)
                        )

            mvb4menu = menu.addMenu("Move after")
            for oa in container.actions:
                if oa.context_key is container.current_key and oa not in acts:
                    mvb4menu.addAction(oa.name).triggered.connect(cb_mvaft_gen(acts, oa))

            # TODO: change to drag&drop, mime data using indexes
            
            menu.addSeparator()
            menu.addAction("Delete").triggered.connect(cb_del_gen(acts))

        menu.exec(self.viewport().mapToGlobal(pos))
    
    def action_item_clicked(self, item: QTreeWidgetItem, col: int):
        act = item.data(NAME, Qt.ItemDataRole.UserRole)
        # if not isinstance(act, ActionNode):
        #     return
        self.sig_edit_action_requested.emit(act)

    def action_item_dblclicked(self, item: QTreeWidgetItem, col: int):
        self.run_action( item.data(NAME, Qt.ItemDataRole.UserRole) )

    def action_apply_node_config(self, node: ActionNode, config: dict, fire: bool=False):
        if not isinstance(node, ActionNode):
            return
        
        # if node from previous container, seems works too
        node.apply_construct_config(config)

        node.status = ActionNode.ActionStatus.CONFIGURED
        if fire:
            self.run_action(node)
            # refresh included in `run_action`
        else:
            self.refresh()

class NodeEditorWidget(QWidget):
    sig_return_node = QtCore.pyqtSignal(NodeBase, dict, bool)

    def __init__(self, parent: MainWindowBase):
        super().__init__(parent)

        vlayout = QtWidgets.QVBoxLayout(self)
        vlayout.setContentsMargins(0, 0, 0, 0)

        self.editor = editor = QsciScintilla(self)
        lexer = QsciLexerYAML(editor)
        lexer.setFont(QtGui.QFont("Consolas"))
        editor.setLexer(lexer)
        editor.setUtf8(True)
        editor.setAutoIndent(True)
        # editor.setEolVisibility(True)
        editor.setIndentationGuides(True)
        editor.setTabWidth(4)
        editor.setIndentationsUseTabs(False)
        editor.setMarginType(1, QsciScintilla.NumberMargin)

        btn_layout = QtWidgets.QHBoxLayout()
        apply_btn = QtWidgets.QToolButton(self)
        apply_btn.setText("✔")
        apply_btn.setToolTip("Apply config")
        fire_btn = QtWidgets.QToolButton(self)
        fire_btn.setText("🔥")
        fire_btn.setToolTip("Fire = apply + run")
        apply_btn.clicked.connect(self.action_apply)
        fire_btn.clicked.connect(partial(self.action_apply, fire=True))

        vlayout.addWidget(editor)
        btn_layout.addStretch(1)
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(fire_btn)
        vlayout.addLayout(btn_layout)

        self._current_node = None

    def edit_node(self, node: NodeBase):
        s = yaml.dump(node.get_construct_config(), allow_unicode=True)
        self.editor.setText(s + "\n# " + type(node).__name__)
        self._current_node = node

    def action_apply(self, fire=True):
        if self._current_node is None:
            return
        config = yaml.load(StringIO(self.editor.text()), Loader=yaml.FullLoader)
        self.sig_return_node.emit(self._current_node, config, fire)

if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindowBase()
    win.show()
    app.exit(app.exec())