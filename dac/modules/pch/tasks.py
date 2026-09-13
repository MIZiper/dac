"""Quick tasks for PCH module.

Provides ``SetupAnalysisContextTask`` that creates a new analysis context
from a time range selected interactively via ``SelectTimeRangeAction``.

The built-in task creates a ``LoadAndCropAction``.  Downstream apps that
need a different load pattern can subclass and override
:meth:`SetupAnalysisContextTask.build_load_actions`.
"""

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt

from dac.core import ContextKeyNode
from dac.core.actions import ActionBase
from dac.core.data import SimpleDefinition
from dac.gui import TaskBase
from . import TimeChannel, time_to_str
from .actions import LoadAndCropAction, SelectTimeRangeAction
from .plots import STATISTICS


_CMAPS = ["jet", "viridis", "plasma", "inferno", "magma", "turbo", "coolwarm"]


def _parse_bins(text: str):
    """Parse "start, end, step" into a 3-float list, or ``None`` when blank."""
    text = (text or "").strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("expected 'start, end, step'")
    return [float(p) for p in parts]


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

        # --- check all / uncheck all ---
        toggle_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Check all")
        btn_none = QtWidgets.QPushButton("Uncheck all")
        btn_all.clicked.connect(lambda: self._set_all_checked(Qt.CheckState.Checked))
        btn_none.clicked.connect(lambda: self._set_all_checked(Qt.CheckState.Unchecked))
        toggle_row.addWidget(btn_all)
        toggle_row.addWidget(btn_none)
        toggle_row.addStretch()
        layout.addLayout(toggle_row)

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

    def _set_all_checked(self, state: Qt.CheckState):
        for i in range(self._list_widget.count()):
            self._list_widget.item(i).setCheckState(state)

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
    load action(s) in that context.

    The built-in :meth:`build_context` creates a ``SimpleDefinition``
    context key and a single ``LoadAndCropAction``.  Downstream apps that
    need a different context key type or load pattern subclass this task
    and override that one method.

    On construction the task installs itself as the *setup handler* on
    ``SelectTimeRangeAction`` so the canvas right-click can trigger it.
    """

    def __init__(self, dac_win: "MainWindow", name: str, *args):
        super().__init__(dac_win, name, *args)
        SelectTimeRangeAction.setup_handler = self

    def build_context(
        self,
        context_name: str,
        channels: list[TimeChannel],
        fpaths: list[str],
        t_start,
        t_end,
    ) -> tuple[ContextKeyNode, list[ActionBase]]:
        """Build the context key and its load action(s). Override to customize.

        Returns ``(context_key, actions)``; both are configured but not
        yet added to the container — the caller registers the key and
        appends the actions under it.
        """
        is_point = t_start is not None and t_start == t_end
        context_key = SimpleDefinition(name=context_name)
        act = LoadAndCropAction(context_key=context_key)
        act.get_construct_config()
        act._construct_config.update(
            {
                "fpaths": list(fpaths),
                "t_start": time_to_str(t_start),
                "t_end": time_to_str(t_end if not is_point else None),
            }
        )
        return context_key, [act]

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

        # build context key + load action(s) — overridable
        new_key, actions = self.build_context(
            context_name, selected_channels, list(fpaths), t_start, t_end
        )
        container.context_keys.add_node(new_key)
        for act in actions or []:
            container.actions.append(act)

        # refresh the UI
        self.dac_win.data_list_widget.refresh()
        self.dac_win.action_list_widget.refresh()
        self.dac_win.message(
            f"Created context '{context_name}' with {len(fpaths)} file(s)"
        )


# ---------------------------------------------------------------------------
# XYStatisticTask — configure the XY statistic color plot
# ---------------------------------------------------------------------------


class XYStatisticDialog(QtWidgets.QDialog):
    """Modal dialog to configure an :class:`XYStatisticPlotAction`.

    The clicked channel is fixed as x; the user picks y, an optional z,
    the averaging window, bin specs, and the statistic.
    """

    def __init__(
        self,
        x_channel: TimeChannel,
        channels: list[TimeChannel],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("XY statistic color plot")
        self.setMinimumWidth(420)

        self._x_channel = x_channel
        self._channels = channels
        self._result = None

        form = QtWidgets.QFormLayout(self)

        form.addRow("X channel:", QtWidgets.QLabel(x_channel.name))

        self._y_combo = QtWidgets.QComboBox()
        for ch in channels:
            self._y_combo.addItem(ch.name, ch)
        for i in range(self._y_combo.count()):
            if self._y_combo.itemData(i) is not x_channel:
                self._y_combo.setCurrentIndex(i)
                break
        form.addRow("Y channel:", self._y_combo)

        self._z_combo = QtWidgets.QComboBox()
        self._z_combo.addItem("<None>", None)
        for ch in channels:
            self._z_combo.addItem(ch.name, ch)
        form.addRow("Z channel:", self._z_combo)

        self._avg_spin = QtWidgets.QDoubleSpinBox()
        self._avg_spin.setRange(0.001, 1e9)
        self._avg_spin.setValue(1.0)
        self._avg_spin.setSuffix(" s")
        form.addRow("Average window:", self._avg_spin)

        self._x_bins_edit = QtWidgets.QLineEdit()
        self._x_bins_edit.setPlaceholderText("start, end, step  (blank = auto)")
        form.addRow("X bins:", self._x_bins_edit)

        self._y_bins_edit = QtWidgets.QLineEdit()
        self._y_bins_edit.setPlaceholderText("start, end, step  (blank = auto)")
        form.addRow("Y bins:", self._y_bins_edit)

        self._stat_combo = QtWidgets.QComboBox()
        self._stat_combo.addItems(list(STATISTICS))
        form.addRow("Statistic:", self._stat_combo)

        self._cmap_combo = QtWidgets.QComboBox()
        self._cmap_combo.addItems(_CMAPS)
        form.addRow("Colormap:", self._cmap_combo)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _on_accept(self):
        try:
            x_bins = _parse_bins(self._x_bins_edit.text())
            y_bins = _parse_bins(self._y_bins_edit.text())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Invalid bins", str(e))
            return

        y_ch = self._y_combo.currentData()
        if y_ch is None:
            QtWidgets.QMessageBox.warning(self, "Missing channel", "Pick a Y channel.")
            return

        self._result = {
            "y_channel": y_ch.name,
            "z_channel": (self._z_combo.currentData().name
                          if self._z_combo.currentData() is not None else None),
            "avg_seconds": self._avg_spin.value(),
            "x_bins": x_bins,
            "y_bins": y_bins,
            "statistic": self._stat_combo.currentText(),
            "cmap": self._cmap_combo.currentText(),
        }
        self.accept()

    def result(self):
        return self._result


class XYStatisticTask(TaskBase):
    """DEFAULT_TASK for ``XYStatisticPlotAction`` (flash quick action).

    The channel the user right-clicked is already stored as ``x_channel``
    in the action config; this task collects the remaining parameters.
    """

    def __call__(self, action: ActionBase):
        container = self.dac_win.container
        if container is None:
            return

        channels = list(container.CurrentContext.nodes_of_type(TimeChannel))
        if not channels:
            QtWidgets.QMessageBox.information(
                self.dac_win, "No channels", "No TimeChannels in the current context."
            )
            return

        raw_x = action._construct_config.get("x_channel")
        x_channel = raw_x
        if isinstance(raw_x, str):
            x_channel = container.CurrentContext.get_node_of_type(raw_x, TimeChannel)
        if x_channel is None:
            x_channel = channels[0]

        dlg = XYStatisticDialog(x_channel, channels, parent=self.dac_win)
        if not dlg.exec_():
            return

        action._construct_config.update(dlg.result())
        return True


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
