"""Defines GUI quick tasks related to TimeData for the DAC application.

Quick-task is used to help action node configuration.
"""

from os import path
from PyQt5 import QtWidgets, QtCore
from dac.core.actions import ActionBase
from dac.gui import TaskBase
from dac import APPNAME

APPSETTING = QtCore.QSettings(APPNAME, "TimeData")
SET_RECENTDIR = "RecentDir"

class FillFpathsTask(TaskBase):
    def __call__(self, action: ActionBase):
        """Opens a file dialog for the user to select measurement files.

        Parameters
        ----------
        action : ActionBase
            The ActionBase instance who has key 'fpaths'.
        """

        fpaths, fext = QtWidgets.QFileDialog.getOpenFileNames(
            self.dac_win, caption="Select measurement files",
            directory=APPSETTING.value(SET_RECENTDIR)
        )
        if not fpaths:
            return
        APPSETTING.setValue(SET_RECENTDIR, path.dirname(fpaths[0]))

        action._construct_config['fpaths'] = fpaths