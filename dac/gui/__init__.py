import importlib
import inspect
import json
import re
import sys
from functools import partial
from glob import glob
from io import BytesIO, StringIO
from os import path

import yaml
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
                                                NavigationToolbar2QT)
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qsci import QsciLexerPython, QsciLexerYAML, QsciScintilla
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent, QMouseEvent
from PyQt5.QtWidgets import (QMainWindow, QStyle, QTreeWidget, QTreeWidgetItem,
                             QWidget)

from dac import APPNAME, __version__
from dac.core import GCK, ActionNode, Container, DataNode, NodeBase
from dac.core.actions import PAB, VAB, ActionBase
from dac.core.thread import ThreadWorker
from dac.gui.base import MainWindowBase

NAME, TYPE, REMARK = range(3)
SET_RECENTDIR = "RecentDir"



class TaskBase:
    def __init__(self, dac_win: "MainWindow", name: str, *args):
        self.dac_win = dac_win
        self.name = name

    def request_update_action(self):
        pass

    def __call__(self, action: ActionBase):
        pass

class MainWindow(MainWindowBase):
    APPTITLE = APPNAME
    APPSETTING = QtCore.QSettings(APPNAME, "Main")

    def __init__(self) -> None:
        super().__init__()
        self.resize(1200, 800)

        self.figure: Figure = None

        self._create_ui()
        self._create_menu()
        self._create_status()
        self._route_signals()

        self.container: Container = None
        self.project_config_fpath: str = None
        self.apply_config({})
        
    def _create_ui(self):
        super()._create_ui()

        self.setDockNestingEnabled(True)
        self.data_list_widget = data_list = DataListWidget(self)
        self.action_list_widget = action_list = ActionListWidget(self)
        self.node_editor = node_editor = NodeEditorWidget(self)

        data_list_docker = QtWidgets.QDockWidget("Data", self)
        data_list_docker.setWidget(data_list)
        action_list_docker = QtWidgets.QDockWidget("Action", self)
        action_list_docker.setWidget(action_list)
        node_editor_docker = QtWidgets.QDockWidget("Editor", self)
        node_editor_docker.setWidget(node_editor)

        self.figure = figure = Figure()
        self.canvas = canvas = FigureCanvasQTAgg(figure)
        self.navibar = navibar = NavigationToolbar2QT(canvas, self)

        canvas.mpl_connect("key_press_event", lambda event: key_press_handler(event, canvas, navibar))
        canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        center_widget = QtWidgets.QWidget(self)
        vlayout = QtWidgets.QVBoxLayout(center_widget)
        vlayout.addWidget(canvas)
        vlayout.addWidget(navibar)

        self.setCentralWidget(center_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, data_list_docker)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, node_editor_docker)
        self.splitDockWidget(data_list_docker, action_list_docker, Qt.Orientation.Horizontal)
    
    def _create_menu(self):
        super()._create_menu()
        
        menubar = self.menuBar()
        app_menu = menubar.addMenu("&App")

        new_project_action = app_menu.addAction("&New project")
        app_menu.addSeparator()
        save_project_action = app_menu.addAction("&Save project")
        saveas_project_action = app_menu.addAction("Save as ...")
        load_project_action = app_menu.addAction("&Load project")
        app_menu.addSeparator()
        exit_action = app_menu.addAction("E&xit")

        def action_new_project():
            self.project_config_fpath = None
            self.apply_config({})
            self.message("New project created")
        def action_save():
            config_fpath = self.project_config_fpath
            if config_fpath is None:
                action_saveas()
                return
            with open(config_fpath, mode="w", encoding="utf8") as fp:
                config = self.get_config()
                json.dump(config, fp, indent=2)
                self.message(f"Save project to {config_fpath}")
        def action_saveas():
            fpath, fext = QtWidgets.QFileDialog.getSaveFileName(
                parent=self, caption="Save project configuration", filter="DAC config (*.dac.json);;All(*.*)",
                directory=self.APPSETTING.value(SET_RECENTDIR)
            )
            if not fpath:
                return
            self.APPSETTING.setValue(SET_RECENTDIR, path.dirname(fpath))
            self.project_config_fpath = fpath
            action_save()
            self.setWindowTitle(f"{path.basename(fpath)} | {self.APPTITLE}")
        def action_load_project():
            fpath, fext = QtWidgets.QFileDialog.getOpenFileName(
                parent=self, caption="Open project configuration", filter="DAC config (*.json);;All (*.*)",
                directory=self.APPSETTING.value(SET_RECENTDIR)
            )
            if not fpath:
                return
            self.APPSETTING.setValue(SET_RECENTDIR, path.dirname(fpath))
            with open(fpath, mode="r", encoding="utf8") as fp:
                config = json.load(fp)
            self.project_config_fpath = fpath
            self.apply_config(config)
            self.message(f"Project loaded from {fpath}")

        new_project_action.triggered.connect(action_new_project)
        save_project_action.triggered.connect(action_save)
        saveas_project_action.triggered.connect(action_saveas)
        load_project_action.triggered.connect(action_load_project)
        exit_action.triggered.connect(self.close)

        tool_menu = self._dac_menu
        copy_action = tool_menu.addAction("Copy figure", self.action_copy_figure)
        tool_menu.insertAction(tool_menu.actions()[0], copy_action)
        tool_menu.addAction("Reload modules", self.action_reload_modules, shortcut=Qt.CTRL+Qt.Key_R)

        # TODO: debug mode {no thread; show action code; send data to ipy; reload modules}

        menubar.addMenu(self._dac_menu)
    
    def _create_status(self):
        return super()._create_status()

    def _route_signals(self):
        self.data_list_widget.sig_edit_data_requested.connect(self.node_editor.edit_node)
        self.action_list_widget.sig_edit_action_requested.connect(self.node_editor.edit_node)
        self.data_list_widget.sig_action_update_requested.connect(
            self.action_list_widget.refresh
        )
        self.data_list_widget.sig_action_runall_requested.connect(
            self.action_list_widget.run_all_actions
        )
        self.action_list_widget.sig_data_update_requested.connect(
            self.data_list_widget.refresh
        )
        self.node_editor.sig_return_node.connect(self.data_list_widget.action_apply_node_config)
        self.node_editor.sig_return_node.connect(self.action_list_widget.action_apply_node_config)
        
    def action_copy_figure(self):
        if self.figure is None:
            return
        with BytesIO() as buf:
            self.figure.savefig(buf)
            QtWidgets.QApplication.clipboard().setImage(QtGui.QImage.fromData(buf.getvalue()))

    def action_reload_modules(self):
        dlg = MultipleItemsDialog(self, sorted([mod.__name__ for mod in Container._modules]),
                                  caption="Reload modules", label="Select modules which need reloaded. \nFor function dev purpose. \nIn some cases, reloading project is required to make the change effect.")
        dlg.exec()
        if dlg.result() and (itms := dlg.get_result()):
            for mod_name in itms:
                mod = importlib.import_module(mod_name)
                importlib.reload(mod)
                # the sequence may be an issue, e.g. ModA depend on ModB, but ModA get reloaded first, and then ModB. (Then reload it twice?)

    def spawn_cofigure(self):
        figure = Figure()
        canvas = FigureCanvasQTAgg(figure)
        navibar = NavigationToolbar2QT(canvas, self)

        canvas.mpl_connect("key_press_event", lambda event: key_press_handler(event, canvas, navibar))
        canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        widget = QtWidgets.QWidget(self, flags=Qt.WindowType.Tool)
        widget.setWindowTitle("Copilot figure")
        widget.resize(650, 450)
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(navibar)
        layout.addWidget(canvas)
        widget.show()

        return figure

    def use_plugins(self, setting_fpath: str, clean: bool=True):
        alias_pattern = re.compile("^/(?P<alias_name>.+)/(?P<rest>.+)")
        def get_node_type(cls_path: str) -> str | type[NodeBase]:
            if cls_path[0]=="[" and cls_path[-1]=="]":
                return cls_path # just str as section string
            
            if (rst:=alias_pattern.search(cls_path)):
                cls_path = alias[rst['alias_name']]+"."+rst['rest']

            try:
                return Container.GetClass(cls_path)
            except AttributeError:
                self.message(f"Module `{cls_path}` not found")
                return None
            
        if clean:
            Container._context_action_types.clear()
            Container._global_node_types.clear()
            # quick_tasks and quick_actions are always overwritten

        with open(setting_fpath, mode="r", encoding="utf8") as fp:
            setting: dict = yaml.load(fp, Loader=yaml.FullLoader)
            if not setting: return

            if (inherit_rel_path:=setting.get('inherit')) is not None:
                self.use_plugins(path.join(path.dirname(setting_fpath), inherit_rel_path), clean=False)

            alias = setting.get('alias', {})

            for gdts in setting.get('data', {}).get("_", []): # global_data_type_string
                node_type = get_node_type(gdts)
                if node_type: Container.RegisterGlobalDataType(node_type)

            for dts, catss in setting.get('actions', {}).items(): #  data_type_string, context_action_type_string_s
                if dts=="_": # global_context
                    for cats in catss:
                        node_type = get_node_type(cats)
                        if node_type: Container.RegisterGlobalContextAction(node_type)
                else:
                    data_type = get_node_type(dts)
                    if not data_type: continue
                    for cats in catss:
                        action_type = get_node_type(cats)
                        if action_type: Container.RegisterContextAction(data_type, action_type)

            for ats, tss in setting.get("quick_tasks", {}).items(): # action_type_string, task_string_s
                action_type = get_node_type(ats)
                if not action_type: continue
                action_type.QUICK_TASKS = [] # make superclass.QUICK_TASKS hidden
                for tts, name, *rest in tss: # task_type_string, name, *rest
                    task_type = get_node_type(tts)
                    if not task_type: continue
                    task = task_type(dac_win=self, name=name, *rest)
                    action_type.QUICK_TASKS.append(task)

            for dts, ass in setting.get("quick_actions", {}).items(): # data_type_string, action_string_s
                data_type = get_node_type(dts)
                if not data_type: continue
                data_type.QUICK_ACTIONS = []
                for ats, dpn, opd in ass: # action_type_string, data_param_name, other_params_dict
                    action_type = get_node_type(ats)
                    if not action_type: continue
                    data_type.QUICK_ACTIONS.append((action_type, dpn, opd))

    def use_plugins_dir(self, setting_dpath: str, default: str=None):
        menubar = self.menuBar()
        plugin_menu = menubar.addMenu("Plugins") # TODO: make it about-to-open style, so search files everytime

        def apply_plugins_file_gen(f):
            def apply_plugins_file():
                self.use_plugins(f, clean=True)
            return apply_plugins_file
        
        for f in glob(path.join(setting_dpath, "*.yaml")):
            act = plugin_menu.addAction(path.basename(f))
            act.triggered.connect(apply_plugins_file_gen(f))

        if default:
            self.use_plugins(path.join(setting_dpath, default))

    def apply_config(self, config: dict):
        if self.project_config_fpath:
            self.setWindowTitle(f"{path.basename(self.project_config_fpath)} | {self.APPTITLE}")
        else:
            self.setWindowTitle(f"[New project] | {self.APPTITLE}")

        dac_config = config.get("dac", {})
        self.container = container = Container.parse_save_config(dac_config)
        self.data_list_widget.refresh(container)
        self.action_list_widget.refresh(container)

    def get_config(self):
        return {
            "_": {"version": __version__},
            "dac": {} if self.container is None else self.container.get_save_config()
        }

class DataListWidget(QTreeWidget):
    sig_edit_data_requested = QtCore.pyqtSignal(DataNode)
    sig_action_update_requested = QtCore.pyqtSignal()
    sig_action_runall_requested = QtCore.pyqtSignal()
    
    def __init__(self, parent: MainWindow) -> None:
        super().__init__(parent)
        self._STYLE = self.style()
        self.dac_win = parent
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
            itm.setText(REMARK, node_object.uuid)
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
                itm.setText(REMARK, node_object.uuid)
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
                # if type(node) is type(node_object): # or subclass?
                #     nodes.append(node)
                nodes.append(node)

            def cb_quickaction_gen(qat: tuple[type[ActionBase], str, dict], data_nodes: list[DataNode]):
                act_type, data_param_name, other_params = qat
                def cb_quickaction():
                    params = {data_param_name: data_nodes, **other_params}
                    act = act_type(context_key=container.current_key)
                    act.container = container
                    if isinstance(act, VAB):
                        act.figure = self.dac_win.figure
                    act.pre_run()
                    act(**params)
                    act.post_run()
                return cb_quickaction
            
            for qat  in node_object.QUICK_ACTIONS:
                qat: tuple[type[ActionBase], str, dict]
                act_type, data_param_name, other_params = qat
                menu.addAction(act_type.CAPTION).triggered.connect(cb_quickaction_gen(qat, nodes))
            menu.addSeparator()

        def cb_pushnode_gen(key_object):
            def cb_pushnode():
                self.dac_win.action_toggle_ipy_widget(dac_node=key_object)

            return cb_pushnode
        
        if (uneditable:=itm.data(TYPE, Qt.ItemDataRole.UserRole)): # data nodes in local context
            menu.addAction("Push to IPy").triggered.connect(cb_pushnode_gen(node_object))
            # TODO: enable delete local object
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
            
            def cb_activaterun_gen(key_object):
                def cb_activaterun():
                    cb_activate_gen(key_object)()
                    self.sig_action_runall_requested.emit()
                return cb_activaterun

            menu.addAction("Activate and Run-all").triggered.connect(cb_activaterun_gen(node_object))
            
            if node_object is container.current_key:
                menu.addAction("De-activate").triggered.connect(cb_activate_gen(GCK))
            else:
                menu.addAction("Activate").triggered.connect(cb_activate_gen(node_object))

            def cb_del_gen(key_object):
                def cb_del():
                    if key_object is container.current_key:
                        container.activate_context(GCK)
                        self.sig_action_update_requested.emit()

                    container.remove_global_node(key_object)
                    self.refresh()

                return cb_del
            
            menu.addSeparator()
            menu.addAction("Push to IPy").triggered.connect(cb_pushnode_gen(node_object))
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

    def __init__(self, parent: MainWindow) -> None:
        super().__init__(parent)
        self._STYLE = self.style()
        self.dac_win = parent
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

    def run_action(self, action: ActionNode, complete_cb: callable=None):
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
            if callable(complete_cb):
                complete_cb()

        action.container = container
        self.dac_win.message(f"[{action.name}]")

        if isinstance(action, VAB):
            action.figure = self.dac_win.figure

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
            self.dac_win.start_thread_worker(worker)
        else:
            action.pre_run()
            rst = action(**params)
            action.post_run()
            completed(rst)

    def run_all_actions(self): # both `container.get_node_of_type` and `current_context.add_node(rst)` need current context
        if (container := self._container) is None:
            return
        
        def action_yield():
            for action in container.ActionsInCurrentContext:
                if isinstance(action, VAB):
                    continue
                yield action

        action_yielder = action_yield()
        
        def run_next_action():
            try:
                action = next(action_yielder)
                self.run_action(action, run_next_action)
            except StopIteration:
                pass

        run_next_action()
        
    def action_context_requested(self, pos: QtCore.QPoint):
        if (container := self._container) is None:
            return
        itms = self.selectedItems()
        menu = QtWidgets.QMenu("ActionMenu")

        def add_new_actions(menu: QtWidgets.QMenu, index: int=None):
            menu_stack = []
            def cb_creation_gen(a_t: type[ActionNode]):
                def cb_creation():
                    a = a_t(context_key=container.current_key)
                    if index is None:
                        container.actions.append(a)
                    else:
                        container.actions.insert(index, a)
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
        if not itms:
            add_new_actions(menu)
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
                submenu = menu.addMenu("Insert before")
                add_new_actions(submenu, container.actions.index(act))

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
                    
            def cb_showcode_gen(a: ActionNode):
                def cb_showcode():
                    try:
                        src = inspect.getsource(a.__class__)
                    except ModuleNotFoundError: # in compiled program, no src code
                        self.dac_win.message("No src code available", log=False)
                        return
                    editor = QsciScintilla(self.dac_win)
                    editor.setWindowFlag(Qt.WindowType.Tool)
                    editor.resize(666, 333)
                    lexer = QsciLexerPython(editor)
                    lexer.setFont(QtGui.QFont("Consolas"))
                    editor.setLexer(lexer)
                    editor.setUtf8(True)
                    editor.setAutoIndent(True)
                    # editor.setEolVisibility(True)
                    editor.setIndentationGuides(True)
                    editor.setTabWidth(4)
                    editor.setIndentationsUseTabs(False)
                    editor.setMarginType(1, QsciScintilla.NumberMargin)

                    editor.setWindowTitle(a.name)
                    editor.setText(src)
                    editor.show()
                return cb_showcode
            
            menu.addSeparator()
            menu.addAction("Show code").triggered.connect(cb_showcode_gen(acts[0]))
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

    def __init__(self, parent: MainWindow):
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
        s = yaml.dump(node.get_construct_config(), allow_unicode=True, sort_keys=False)
        self.editor.setText(s + "\n# " + type(node).__name__)
        self._current_node = node

    def action_apply(self, fire=True):
        if self._current_node is None:
            return
        config = yaml.load(StringIO(self.editor.text()), Loader=yaml.FullLoader)
        self.sig_return_node.emit(self._current_node, config, fire)


class MultipleItemsDialog(QtWidgets.QDialog):
    def __init__(self, parent: QWidget, items: list[str], caption: str, label: str) -> None:
        super().__init__(parent)
        self._items = items

        self.setWindowTitle(caption)
        self.item_list = item_list = QtWidgets.QListWidget(self)
        item_list.setSelectionMode(item_list.SelectionMode.ExtendedSelection)
        item_list.addItems(items)
        cap_label = QtWidgets.QLabel(label, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(cap_label)
        layout.addWidget(item_list)
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        layout.addWidget(btn_box)

        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

    def get_result(self):
        return [itm.text() for itm in self.item_list.selectedItems()]

if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.use_plugins_dir(path.join(path.dirname(__file__), "../plugins"), default="0.base.yaml")
    win.show()
    app.exit(app.exec())