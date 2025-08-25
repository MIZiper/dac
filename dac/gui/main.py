"""Main entry point for desktop DAC application.
"""

import json, sys, click
from os import path

from PyQt5 import QtWidgets
from dac.gui import MainWindow

@click.command()
@click.option("--project-file", help="Project file to load")
@click.option("--scenario-file", help="YAML file for scenarios")
def main(project_file: str, scenario_file: str):
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()

    # add splash progress for module loading

    if project_file is not None:
        with open(project_file, mode="r") as fp:
            config = json.load(fp)
            win.apply_config(config)

    # setting_fpath = path.join(path.dirname(__file__), "..", "scenarios/0.base.yaml")
    # win.use_scenario(setting_fpath)
    win.use_scenarios_dir(path.join(path.dirname(__file__), "../scenarios"), default="0.base.yaml")

    win.show()
    app.exit(app.exec())

if __name__=="__main__":
    main()