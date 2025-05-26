"""Defines GUI tasks related to time-series data for the DAC application.

This module provides `FillFpathsTask`, a `TaskBase` subclass that opens a
file dialog for the user to select measurement files. The selected file paths
are then used to configure an action, typically for loading time-series data.
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

        This task is typically used to populate the `fpaths` (file paths)
        argument of an action (e.g., a data loading action). It remembers
        the last used directory for convenience.

        Parameters
        ----------
        action : ActionBase
            The ActionBase instance whose `_construct_config` will be
            updated with the selected file paths under the key 'fpaths'.
        """
        fpaths, fext = QtWidgets.QFileDialog.getOpenFileNames(
            self.dac_win, caption="Select measurement files",
            directory=APPSETTING.value(SET_RECENTDIR)
        )
        if not fpaths:
            return
        APPSETTING.setValue(SET_RECENTDIR, path.dirname(fpaths[0]))

        action._construct_config['fpaths'] = fpaths