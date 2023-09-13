from os import path
import json, yaml, re
import click

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

from dac import __version__, APPNAME
from dac.core import NodeBase, ActionNode, DataNode, Container
from dac.gui import MainWindowBase, DataListWidget, ActionListWidget, NodeEditorWidget

SET_RECENTDIR = "RecentDir"

class MainWindow(MainWindowBase):
    APPTITLE = APPNAME
    APPSETTING = QtCore.QSettings(APPNAME, "Main")

    def __init__(self) -> None:
        super().__init__()
        self.resize(1200, 800)

        self._create_ui()
        self._create_menu()
        self._create_status()
        self._route_signals()

        self.container: Container = None
        self.project_config_fpath = None
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

        # strong focus on canvas

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

    def use_plugins(self, setting_fpath: str):
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

        with open(setting_fpath, mode="r", encoding="utf8") as fp:
            setting: dict = yaml.load(fp, Loader=yaml.FullLoader)

            alias = setting['alias']

            for gdts in setting['data']["_"]: # global_data_type_string
                node_type = get_node_type(gdts)
                if node_type: Container.RegisterGlobalDataType(node_type)

            for dts, catss in setting['actions'].items(): #  data_type_string, context_action_type_string_s
                if dts=="_": # global_context
                    for cats in catss:
                        node_type = get_node_type(cats)
                        if node_type: Container.RegisterGlobalContextAction(node_type)
                else:
                    data_type = get_node_type(dts)
                    if not node_type: continue
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

@click.command()
@click.option("--config-file", help="Configuration file to load")
def main(config_file: str):
    win = MainWindow()
    if config_file is not None:
        with open(config_file, mode="r") as fp:
            config = json.loads(config_file)
            win.apply_config(config)

if __name__=="__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()

    # add splash progress for module loading
    setting_fpath = path.join(path.dirname(__file__), "..", "plugins.yaml")
    win.use_plugins(setting_fpath)

    win.show()
    app.exit(app.exec())