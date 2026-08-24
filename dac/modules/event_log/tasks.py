"""GUI task for interactively adding event log entries.

Provides ``AddEventLogTask`` — a :class:`~dac.gui.TaskBase` registered as
a QUICK_TASK for ``SelectEventRangeAction``.  When triggered (right-click
on the action or on the canvas), it opens a modal dialog that lets the
user pick (or create) a ``CreateEventLogAction`` and enter a label for
the selected time range.
"""

import numpy as np

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from dac.core.actions import ActionBase
from dac.gui import TaskBase
from dac.modules.pch import time_to_str

from .actions import (
    CreateEventLogAction,
    ExtractEventStatisticsAction,
    InspectTimeRangeAction,
    SelectEventRangeAction,
    _compute_stats,
    _nearest_sample,
)

_NEW_GROUP_MARKER = "  [ New Group … ]"


def _group_name_of(act: CreateEventLogAction) -> str:
    """Return the display name of a CreateEventLogAction.

    Uses ``out_name`` when the user set it; falls back to the action name
    when ``out_name`` is still the ``<...>`` placeholder.
    """
    on = act.out_name
    if on and not (on.startswith("<") and on.endswith(">")):
        return on
    return act.name


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class _AddEventLogDialog(QtWidgets.QDialog):
    """Modal dialog for adding one event log entry.

    Shows the selected time range, lets the user pick or type a group
    name, and enter a label string.
    """

    def __init__(self, t_start, t_end, container, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Event Log Entry")
        self.setMinimumWidth(400)

        self._t_start = t_start
        self._t_end = t_end
        self._container = container
        self._result = None

        layout = QtWidgets.QVBoxLayout(self)

        # ---- time range (read-only) ----
        time_row = QtWidgets.QHBoxLayout()
        time_row.addWidget(QtWidgets.QLabel("Time range:"))
        t0_str = time_to_str(t_start)
        t1_str = time_to_str(t_end)
        if t_start == t_end:
            time_label = QtWidgets.QLabel(f"Point at  {t0_str}")
        else:
            time_label = QtWidgets.QLabel(f"{t0_str}  →  {t1_str}")
        time_row.addWidget(time_label)
        time_row.addStretch()
        layout.addLayout(time_row)

        # ---- group selection ----
        layout.addWidget(QtWidgets.QLabel("Event log group:"))

        self._group_combo = QtWidgets.QComboBox()
        existing = self._collect_group_names()
        self._group_combo.addItems(existing)
        self._group_combo.addItem(_NEW_GROUP_MARKER)
        self._group_combo.currentTextChanged.connect(self._on_group_changed)
        layout.addWidget(self._group_combo)

        self._new_group_edit = QtWidgets.QLineEdit()
        self._new_group_edit.setPlaceholderText("Enter new group name …")
        self._new_group_edit.setVisible(False)
        layout.addWidget(self._new_group_edit)

        # Initialise the edit-field visibility; when there are no existing
        # groups the combo defaults to the "new group" marker and the signal
        # above never fires during population.
        self._on_group_changed(self._group_combo.currentText())

        # ---- label ----
        label_row = QtWidgets.QHBoxLayout()
        label_row.addWidget(QtWidgets.QLabel("Label:"))
        self._label_edit = QtWidgets.QLineEdit()
        self._label_edit.setPlaceholderText("e.g. Knock, Rattle, Shift …")
        label_row.addWidget(self._label_edit)
        layout.addLayout(label_row)

        # ---- buttons ----
        btn_row = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_row.accepted.connect(self._on_accept)
        btn_row.rejected.connect(self.reject)
        layout.addWidget(btn_row)

    # -- helpers ------------------------------------------------

    def _collect_group_names(self) -> list[str]:
        names = []
        if self._container is None:
            return names
        for act in self._container.actions:
            if isinstance(act, CreateEventLogAction):
                gname = _group_name_of(act)
                if gname and gname not in names:
                    names.append(gname)
        return names

    def _on_group_changed(self, text: str):
        self._new_group_edit.setVisible(text == _NEW_GROUP_MARKER)

    def _on_accept(self):
        label = self._label_edit.text().strip()
        if self._group_combo.currentText() == _NEW_GROUP_MARKER:
            group_name = self._new_group_edit.text().strip()
        else:
            group_name = self._group_combo.currentText().strip()

        if not group_name:
            QtWidgets.QMessageBox.warning(
                self, "Missing group name",
                "Please select or enter a group name.",
            )
            return
        if not label:
            QtWidgets.QMessageBox.warning(
                self, "Missing label", "Please enter a label for the event.",
            )
            return

        self._result = (group_name, label)
        self.accept()

    def result(self):
        return self._result


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class AddEventLogTask(TaskBase):
    """QUICK_TASK for ``SelectEventRangeAction``.

    On construction the task installs itself as the *setup handler* on
    ``SelectEventRangeAction`` so the canvas right-click triggers it.
    """

    def __init__(self, dac_win: "MainWindow", name: str, *args):
        super().__init__(dac_win, name, *args)
        SelectEventRangeAction.setup_handler = self

    def __call__(self, action: ActionBase):
        container = self.dac_win.container
        if container is None:
            return

        t_start = getattr(action, "_t_start", None)
        t_end = getattr(action, "_t_end", None)

        if t_start is None:
            QtWidgets.QMessageBox.information(
                self.dac_win,
                "No selection",
                "Run the action first and select a time range on the plot.",
            )
            return

        dlg = _AddEventLogDialog(t_start, t_end, container, parent=self.dac_win)
        if not dlg.exec():
            return

        group_name, label = dlg.result()
        time_str = f"{time_to_str(t_start)} ~ {time_to_str(t_end)}"

        target: CreateEventLogAction | None = None
        for act in container.actions:
            if isinstance(act, CreateEventLogAction):
                if _group_name_of(act) == group_name:
                    target = act
                    break

        if target is None:
            target = CreateEventLogAction(context_key=action.context_key)
            target.container = container
            target.get_construct_config()
            target.out_name = group_name
            target._construct_config["event_data"] = []
            target.status = CreateEventLogAction.ActionStatus.CONFIGURED
            container.actions.append(target)
            self.dac_win.message(f"Created event log group '{group_name}'")

        if not isinstance(target._construct_config.get("event_data"), list):
            target._construct_config["event_data"] = []
        target._construct_config["event_data"].append([time_str, label])
        target.status = CreateEventLogAction.ActionStatus.CONFIGURED

        self.dac_win.action_list_widget.refresh()
        self.dac_win.message(
            f"Added event '{label}' to group '{group_name}'"
        )


# ---------------------------------------------------------------------------
# InspectSelectionTask — quick view of a selected range / point
# ---------------------------------------------------------------------------


class InspectSelectionTask(TaskBase):
    """QUICK_TASK for ``InspectTimeRangeAction``.

    On construction the task installs itself as the *setup handler* on
    ``InspectTimeRangeAction`` so the canvas right-click triggers it.
    """

    def __init__(self, dac_win: "MainWindow", name: str, *args):
        super().__init__(dac_win, name, *args)
        InspectTimeRangeAction.setup_handler = self

    def __call__(self, action: ActionBase):
        t_start = getattr(action, "_t_start", None)
        t_end = getattr(action, "_t_end", None)

        if t_start is None:
            QtWidgets.QMessageBox.information(
                self.dac_win,
                "No selection",
                "Run the action first and select a time range on the plot.",
            )
            return

        channels = getattr(action, "_channels", None) or []
        if not channels:
            QtWidgets.QMessageBox.information(
                self.dac_win,
                "No channels",
                "No channels available to inspect.",
            )
            return

        stats = getattr(action, "_stats", "mean,std,min,max,rms")

        if t_start == t_end:
            table = self._point_table(channels, t_start)
        else:
            table = self._range_table(channels, t_start, t_end, stats)

        self.dac_win.show_stats(table)

    def _range_table(self, channels, t_start, t_end, stats):
        wanted = [s.strip() for s in stats.split(",") if s.strip()]
        wanted = [
            s for s in ExtractEventStatisticsAction._AVAILABLE_STATS if s in wanted
        ]

        rows, data = [], []
        for ch in channels:
            _t, y, _dt = ch.get_merged_data(t_start=t_start, t_end=t_end)
            y_clean = y[~np.isnan(y)]
            if len(y_clean) == 0:
                continue
            rec = _compute_stats(y_clean, wanted)
            rows.append(f"{ch.name} [{ch.y_unit}]")
            data.append([rec.get(s, "") for s in wanted])

        return {
            "title": f"Statistics  {time_to_str(t_start)} → {time_to_str(t_end)}",
            "headers": {"row": rows, "col": wanted},
            "data": data,
        }

    def _point_table(self, channels, t):
        rows, data = [], []
        for ch in channels:
            t_sample, value = _nearest_sample(ch, t)
            rows.append(f"{ch.name} [{ch.y_unit}]")
            if t_sample is None:
                data.append(["", ""])
            else:
                data.append([time_to_str(t_sample), value])

        return {
            "title": f"Values at  {time_to_str(t)}",
            "headers": {"row": rows, "col": ["time", "value"]},
            "data": data,
        }
