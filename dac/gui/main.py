"""Main entry point for desktop DAC application.
"""

import json, sys, click
import os
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
        try:
            with open(project_file, mode="r") as fp:
                config = json.load(fp)
                win.apply_config(config)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading project file: {e}", file=sys.stderr)

    # setting_fpath = path.join(path.dirname(__file__), "..", "scenarios/0.base.yaml")
    # win.use_scenario(setting_fpath)
    scen_dir = os.getenv("SCENARIO_DIR") or path.join(path.dirname(__file__), "../scenarios")
    scen_def = os.getenv("SCENARIO_DEFAULT") or "0.base.yaml"
    win.use_scenarios_dir(scen_dir, default=scen_def)

    win.show()
    app.exit(app.exec())

if __name__=="__main__":
    main()