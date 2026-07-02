"""Quick tasks for PCH module.

Provides ``SetupAnalysisContextTask`` that creates a new analysis context
with a ``LoadAndCropAction`` configured to extract data from the time range
selected interactively via ``SelectTimeRangeAction``.
"""

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt

from dac.core.actions import ActionBase
from dac.core.data import SimpleDefinition
from dac.gui import TaskBase
from . import TimeChannel, TimeSegment
from .actions import LoadAndCropAction


class SetupAnalysisDialog(QtWidgets.QDialog):
    """Modal dialog to configure an analysis context.

    Shows the time selection mode (point / range), lets the user name
    the new context, and pick which TimeChannels to include.
    """

    def __init__(
        self,
        t_start,
        t_end,
        channels: list[TimeChannel],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Setup Analysis Context")
        self.setMinimumWidth(420)

        self._channels = channels
        self._t_start = t_start
        self._t_end = t_end
        self._result = None

        layout = QtWidgets.QVBoxLayout(self)

        # --- context name ---
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Context name:"))
        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.setPlaceholderText("e.g. RampUp_50Nm")
        name_row.addWidget(self._name_edit)
        layout.addLayout(name_row)

        # --- selection mode (read-only info) ---
        is_point = t_start is not None and t_start == t_end
        if t_start is not None and t_end is not None:
            if is_point:
                mode_text = f"Point at  {_time_repr(t_start)}"
                detail = "(load full files containing this time)"
            else:
                mode_text = (
                    f"Range  {_time_repr(t_start)}  \u2192  {_time_repr(t_end)}"
                )
                detail = "(crop data to this range)"
        else:
            mode_text = "No time selected yet"
            detail = "Run SelectTimeRangeAction first and select a range"

        mode_group = QtWidgets.QGroupBox("Selection")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        mode_layout.addWidget(QtWidgets.QLabel(mode_text))
        mode_layout.addWidget(QtWidgets.QLabel(f"<i>{detail}</i>"))
        layout.addWidget(mode_group)

        # --- channel checklist ---
        ch_label = QtWidgets.QLabel(
            "Channels overlapping selection:  (check to include)"
        )
        layout.addWidget(ch_label)

        self._list_widget = QtWidgets.QListWidget()
        for ch in channels:
            segs = ch.segments_at(t_start, t_end) if t_start is not None else ch.segments
            file_count = len(set(s._cache_key[0] for s in segs if s._cache_key))
            item = QtWidgets.QListWidgetItem(
                f"  {ch.name}    [{ch.y_unit}]    ({file_count} file(s))"
            )
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, ch)
            self._list_widget.addItem(item)
        layout.addWidget(self._list_widget)

        # --- buttons ---
        btn_row = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_row.accepted.connect(self._on_accept)
        btn_row.rejected.connect(self.reject)
        layout.addWidget(btn_row)

    def _on_accept(self):
        name = self._name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Missing name", "Please enter a context name.")
            return

        selected = []
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.data(Qt.ItemDataRole.UserRole))
        if not selected:
            QtWidgets.QMessageBox.warning(self, "No channels", "Please select at least one channel.")
            return

        self._result = (name, selected)
        self.accept()

    def result(self):
        return self._result


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class SetupAnalysisContextTask(TaskBase):
    """QUICK_TASK for ``SelectTimeRangeAction``.

    Reads the interactively selected time range, shows
    ``SetupAnalysisDialog`` to collect the context name and channel
    selection, then creates a new ``SimpleDefinition`` context key and
    a ``LoadAndCropAction`` in that context.
    """

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

        # find all TimeChannels in current context that overlap the selection
        all_channels: list[TimeChannel] = list(
            self.current_context.nodes_of_type(TimeChannel)
        )
        matching = [
            ch
            for ch in all_channels
            if ch.segments_at(t_start, t_end)
        ]
        if not matching:
            QtWidgets.QMessageBox.information(
                self.dac_win,
                "No matching channels",
                "No TimeChannels in the current context overlap the selected time range.",
            )
            return

        # show dialog
        dlg = SetupAnalysisDialog(t_start, t_end, matching, parent=self.dac_win)
        if not dlg.exec_():
            return

        context_name, selected_channels = dlg.result()

        # collect unique file paths from selected channels' overlapping segments
        fpaths: set[str] = set()
        for ch in selected_channels:
            for seg in ch.segments_at(t_start, t_end):
                if seg._cache_key:
                    fpaths.add(seg._cache_key[0])

        if not fpaths:
            QtWidgets.QMessageBox.warning(
                self.dac_win,
                "No file paths",
                "Could not determine file paths from the selected channels.",
            )
            return

        # create context key and add to container
        new_key = SimpleDefinition(name=context_name)
        container.context_keys.add_node(new_key)

        # create LoadAndCropAction in the new context
        act = LoadAndCropAction(context_key=new_key)
        act.get_construct_config()
        act._construct_config.update(
            {
                "fpaths": list(fpaths),
                "t_start": t_start,
                "t_end": t_end,
            }
        )
        container.actions.append(act)

        # refresh the UI
        self.dac_win.data_list_widget.refresh()
        self.dac_win.action_list_widget.refresh()
        self.dac_win.message(
            f"Created context '{context_name}' with {len(fpaths)} file(s)"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _time_repr(t) -> str:
    """Human-readable string for a time value (datetime64 or float)."""
    import numpy as np

    if isinstance(t, np.datetime64):
        ts = t.astype("datetime64[ms]").astype(object)
        return str(ts).replace("T", " ")
    try:
        return f"{float(t):.3f} s"
    except (TypeError, ValueError):
        return str(t)
