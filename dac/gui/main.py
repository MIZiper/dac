import json
from os import path

import click
from PyQt5 import QtWidgets
from dac.gui import MainWindow

@click.command()
@click.option("--config-file", help="Configuration file to load")
@click.option("--plugin-file", help="YAML file for plugins")
def main(config_file: str, plugin_file: str):
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

    # setting_fpath = path.join(path.dirname(__file__), "..", "plugins/0.base.yaml")
    # win.use_plugin(setting_fpath)
    win.use_plugins_dir(path.join(path.dirname(__file__), "../plugins"), default="0.base.yaml")

    win.show()
    app.exit(app.exec())