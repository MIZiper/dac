from os import path
import json

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

from dac import __version__
from dac.core import NodeBase, ActionNode, DataNode, Container
from dac.gui import MainWindowBase, DataListWidget, ActionListWidget, NodeEditorWidget

APPNAME = "DAC"
APPSETTING = QtCore.QSettings(APPNAME, "Main")
SET_RECENTDIR = "RecentDir"

class MainWindow(MainWindowBase):
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
        tool_menu = menubar.addMenu("&Tool")

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
                directory=APPSETTING.value(SET_RECENTDIR)
            )
            if not fpath:
                return
            APPSETTING.setValue(SET_RECENTDIR, path.dirname(fpath))
            self.project_config_fpath = fpath
            action_save()
            self.setWindowTitle(f"{path.basename(fpath)} | {APPNAME}")
        def action_load_project():
            fpath, fext = QtWidgets.QFileDialog.getOpenFileName(
                parent=self, caption="Open project configuration", filter="DAC config (*.json);;All (*.*)",
                directory=APPSETTING.value(SET_RECENTDIR)
            )
            if not fpath:
                return
            APPSETTING.setValue(SET_RECENTDIR, path.dirname(fpath))
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
    
    def _create_status(self):
        return super()._create_status()

    def _route_signals(self):
        from dac.core.data import SimpleDefinition
        from dac.core.actions import SimpleAction, SimpleGlobalAction

        Container.RegisterGlobalContextAction(SimpleGlobalAction)
        Container.RegisterGlobalDataType(SimpleDefinition)
        Container.RegisterContextAction(SimpleDefinition, SimpleAction)

        from dac.modules.timedata.construct import SignalConstructAction
        from dac.modules.timedata.actions import ShowTimeDataAction

        Container.RegisterContextAction(SimpleDefinition, SignalConstructAction)
        Container.RegisterContextAction(SimpleDefinition, ShowTimeDataAction)

        self.data_list_widget.sig_edit_data_requested.connect(self.node_editor.edit_node)
        self.action_list_widget.sig_edit_action_requested.connect(self.node_editor.edit_node)
        self.data_list_widget.sig_action_update_requested.connect(
            self.action_list_widget.refresh
        )
        self.action_list_widget.sig_data_update_requested.connect(
            self.data_list_widget.refresh
        )
        self.node_editor.sig_return_node.connect(self.data_list_widget.action_apply_node_config)
        self.node_editor.sig_return_node.connect(self.action_list_widget.action_apply_node_config)

    def apply_config(self, config: dict):
        if self.project_config_fpath:
            self.setWindowTitle(f"{path.basename(self.project_config_fpath)} | {APPNAME}")
        else:
            self.setWindowTitle(f"[New project] | {APPNAME}")

        dac_config = config.get("dac", {})
        self.container = container = Container.parse_save_config(dac_config)
        self.data_list_widget.refresh(container)
        self.action_list_widget.refresh(container)

    def get_config(self):
        return {
            "_": {"version": __version__},
            "dac": {} if self.container is None else self.container.get_save_config()
        }


if __name__=="__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exit(app.exec())