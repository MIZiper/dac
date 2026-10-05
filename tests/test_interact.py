"""Tests for the live interaction layer (`dac.core.interact`)."""

import numpy as np
import pytest

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dac.core import Container, GCK
from dac.core.actions import VAB, PAB
from dac.core.interact import (
    InteractionContext,
    InteractivePlotAction,
    OverlayBase,
    PlotInteractionManager,
    RangeSelectTool,
    ToolBase,
    teardown_manager,
)
from dac.modules.event_log import EventLogCollection
from dac.modules.event_log.interactions import EventRangesOverlay
from dac.modules.pch import TimeChannel, TimeSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _channel(name="ch", t0=0.0, dt=0.1, n=100, unit="g"):
    seg = TimeSegment(name=name, t0=t0, length=n, dt=dt, y_unit=unit)
    seg._y = np.sin(np.linspace(0, 6, n))
    ch = TimeChannel(name=name, y_unit=unit)
    ch.add_segment(seg)
    return ch


class _Event:
    def __init__(self, inaxes=None, xdata=None, ydata=None, button=1):
        self.inaxes = inaxes
        self.xdata = xdata
        self.ydata = ydata
        self.button = button


class _UIStub:
    supports_dialogs = False

    def __init__(self):
        self.stats = None
        self.messages = []
        self.entries = None
        self.cleared = 0

    def message(self, msg, log=True):
        self.messages.append(msg)

    def show_stats(self, stats):
        self.stats = stats

    def install_interactions(self, manager, entries):
        self.entries = entries

    def clear_interactions(self):
        self.cleared += 1


class _BasePlot(VAB):
    def __call__(self, n: int = 10):
        ax = self.figure.gca()
        ax.plot(np.arange(n))
        self.ax = ax


class _DummyOverlay(OverlayBase):
    CAPTION = "Dummy overlay"
    DEFAULT_ON = True
    attached = 0

    def __init__(self, ctx, tag: str = None):
        super().__init__(ctx)
        self.tag = tag

    def attach(self):
        type(self).attached += 1
        self.track(self.ctx.axes[0].axvline(3))


class _DummyTool(ToolBase):
    CAPTION = "Dummy tool"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.presses = 0
        self.attached = False

    def attach(self):
        self.attached = True

    def detach(self):
        self.attached = False
        super().detach()

    def on_press(self, event):
        self.presses += 1


class _DummyUnrelated(PAB):
    def __call__(self):
        return None


class _Host(
    InteractivePlotAction,
    base=_BasePlot,
    interactions=[_DummyOverlay, _DummyTool],
):
    CAPTION = "Host"


def _new_figure():
    fig = Figure()
    FigureCanvasAgg(fig)
    return fig


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


class TestSignatures:
    def test_interaction_signature_from_init(self):
        assert set(_DummyOverlay._SIGNATURE.parameters) == {"tag"}
        assert set(_DummyTool._SIGNATURE.parameters) == set()

    def test_host_signature_is_dict(self):
        assert set(_Host._SIGNATURE) == {"_BasePlot", "_DummyOverlay", "_DummyTool"}
        assert _Host._VALID_PARAM_NAMES == frozenset(_Host._SIGNATURE)

    def test_sequence_exposed_for_quick_actions(self):
        # the GUI quick-action builder relies on `_SEQUENCE` for dict sigs
        assert _Host._SEQUENCE == (_BasePlot, _DummyOverlay, _DummyTool)
        assert "tag" in _Host._SEQUENCE[1]._SIGNATURE.parameters


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class TestManager:
    def _make(self, ui=None):
        container = Container()
        fig = _new_figure()
        fig.add_subplot(111)
        ctx = InteractionContext(fig, container, ui=ui)
        manager = PlotInteractionManager(ctx)
        manager.add(_DummyOverlay(ctx), active=True)
        manager.add(_DummyTool(ctx))
        return manager, fig

    def test_default_overlay_active(self):
        manager, _ = self._make()
        assert manager.is_active("_DummyOverlay")

    def test_set_tool_exclusive(self):
        manager, _ = self._make()
        manager.set_tool("_DummyTool")
        assert manager.is_active("_DummyTool")
        assert manager.active_tool == "_DummyTool"
        manager.set_tool(None)
        assert not manager.is_active("_DummyTool")

    def test_events_routed_to_active_tool_only(self):
        manager, _ = self._make()
        tool = manager.get("_DummyTool")
        manager._dispatch_press(_Event())
        assert tool.presses == 0
        manager.set_tool("_DummyTool")
        manager._dispatch_press(_Event())
        assert tool.presses == 1

    def test_clear_removes_everything_and_notifies_host(self):
        ui = _UIStub()
        manager, _ = self._make(ui=ui)
        manager.clear()
        assert manager.interactions == []
        assert manager._cids == []
        assert ui.cleared == 1

    def test_registry_teardown(self):
        container = Container()
        fig = _new_figure()
        fig.add_subplot(111)
        ctx = InteractionContext(fig, container)
        manager = PlotInteractionManager(ctx)
        from dac.core.interact import install_manager

        install_manager(fig, manager)
        teardown_manager(fig)
        assert manager._cids == []


# ---------------------------------------------------------------------------
# Dialog capability gating
# ---------------------------------------------------------------------------


class TestDialogGating:
    def test_tool_requiring_dialog_unavailable_on_headless(self):
        from dac.modules.pch.interactions import SelectContextTool

        container = Container()
        container.CurrentContext.add_node(_channel(name="AccelX"))
        fig = _new_figure()
        fig.add_subplot(111)
        ctx = InteractionContext(fig, container, ui=_UIStub())
        assert not SelectContextTool.available(ctx)

    def test_tool_requiring_dialog_available_with_dialog_host(self):
        from dac.modules.pch.interactions import SelectContextTool

        class _DialogUI(_UIStub):
            supports_dialogs = True

        container = Container()
        container.CurrentContext.add_node(_channel(name="AccelX"))
        fig = _new_figure()
        fig.add_subplot(111)
        ctx = InteractionContext(fig, container, ui=_DialogUI())
        assert SelectContextTool.available(ctx)


# ---------------------------------------------------------------------------
# Event ranges overlay
# ---------------------------------------------------------------------------


class TestEventRangesOverlay:
    def test_attach_detach_artists(self):
        container = Container()
        coll = EventLogCollection(name="Log")
        coll.add_entry("1.0", "2.5", "Knock")
        container.CurrentContext.add_node(coll)

        fig = _new_figure()
        fig.add_subplot(111)

        ctx = InteractionContext(fig, container)
        overlay = EventRangesOverlay(ctx)
        assert EventRangesOverlay.available(ctx)

        before = len(fig.axes[0].get_children())
        overlay.attach()
        assert len(fig.axes[0].get_children()) > before

        overlay.detach()
        assert len(fig.axes[0].get_children()) == before

    def test_unavailable_without_events(self):
        container = Container()
        ctx = InteractionContext(_new_figure(), container)
        assert not EventRangesOverlay.available(ctx)


# ---------------------------------------------------------------------------
# Range selection tool
# ---------------------------------------------------------------------------


class _StatsTool(RangeSelectTool):
    CAPTION = "Stats"

    def _announce_selection(self):
        self.selection = (self.t_start, self.t_end)


class TestRangeSelectTool:
    def _setup(self, datetime=False):
        container = Container()
        ch = _channel(dt=0.1)
        container.CurrentContext.add_node(ch)
        fig = _new_figure()
        ax = fig.add_subplot(111)
        x = np.arange(100) * 0.1
        if datetime:
            base = np.datetime64("2024-01-01T00:00:00")
            x = base + (np.arange(100) * np.timedelta64(100, "ms"))
        ax.plot(x, np.sin(np.arange(100) * 0.1))
        return container, fig, ax

    def test_range_selection_state(self):
        container, fig, ax = self._setup()
        tool = _StatsTool(InteractionContext(fig, container))
        tool.attach()
        tool.on_press(_Event(inaxes=ax, xdata=1.0))
        tool.on_motion(_Event(inaxes=ax, xdata=3.0))
        tool.on_release(_Event(inaxes=ax, xdata=3.0, button=1))
        assert tool.t_start == pytest.approx(1.0)
        assert tool.t_end == pytest.approx(3.0)
        tool.detach()

    def test_point_selection(self):
        container, fig, ax = self._setup()
        tool = _StatsTool(InteractionContext(fig, container))
        tool.attach()
        tool.on_press(_Event(inaxes=ax, xdata=2.0))
        tool.on_release(_Event(inaxes=ax, xdata=2.0, button=1))
        assert tool.t_start == pytest.approx(2.0)
        assert tool.t_end == pytest.approx(2.0)
        tool.detach()

    def test_datetime_selection_normalized(self):
        import matplotlib.dates as mdates

        container, fig, ax = self._setup(datetime=True)
        tool = _StatsTool(InteractionContext(fig, container))
        tool.attach()
        target = np.datetime64("2024-01-01T00:00:02.000")
        xnum = float(mdates.date2num(target.astype("datetime64[ms]").astype(object)))
        tool.on_press(_Event(inaxes=ax, xdata=xnum))
        tool.on_release(_Event(inaxes=ax, xdata=xnum, button=1))
        assert np.issubdtype(np.asarray(tool.t_start).dtype, np.datetime64)
        tool.detach()


# ---------------------------------------------------------------------------
# pch range statistics tool
# ---------------------------------------------------------------------------


class TestRangeStatsTool:
    def test_presents_stats(self):
        from dac.modules.pch.interactions import RangeStatsTool

        container = Container()
        container.CurrentContext.add_node(_channel(name="AccelX", dt=0.1, unit="g"))
        fig = _new_figure()
        ax = fig.add_subplot(111)
        x = np.arange(100) * 0.1
        ax.plot(x, np.sin(x))

        ui = _UIStub()
        ctx = InteractionContext(fig, container, ui=ui)
        tool = RangeStatsTool(ctx)
        assert RangeStatsTool.available(ctx)
        tool.attach()
        tool.t_start = 1.0
        tool.t_end = 2.0
        tool._announce_selection()
        assert ui.stats is not None
        assert ui.stats["headers"]["row"] == ["mean", "std", "min", "max", "rms"]
        assert ui.stats["headers"]["col"] == ["AccelX"]
        tool.detach()


# ---------------------------------------------------------------------------
# Host end-to-end (headless)
# ---------------------------------------------------------------------------


class TestInteractiveHost:
    def _make_host(self):
        container = Container()
        container.CurrentContext.add_node(_channel(name="AccelX", dt=0.1, unit="g"))
        coll = EventLogCollection(name="Log")
        coll.add_entry("1.0", "2.5", "Knock")
        container.CurrentContext.add_node(coll)

        ui = _UIStub()
        from dac.modules.interactive import InteractiveTimePlotAction

        host = InteractiveTimePlotAction(context_key=GCK)
        host.container = container
        host.ui_host = ui
        host._construct_config["SpecPlotAction"] = {
            "channels": ["AccelX"],
            "spec": None,
        }

        fig = _new_figure()
        host.figure = fig

        params = container.prepare_params_for_action(
            host._SIGNATURE, host._construct_config
        )
        host(**params)
        return host, ui, fig, container

    def test_render_and_install(self):
        host, ui, fig, container = self._make_host()
        manager = host._manager
        names = [type(i).__name__ for i in manager.interactions]
        assert "EventRangesOverlay" in names
        assert "RangeStatsTool" in names
        assert manager.is_active("EventRangesOverlay")
        assert ui.entries is not None
        assert any(
            e.name == "SelectContextTool" and not e.available for e in ui.entries
        )
        assert manager.current_name == "EventRangesOverlay"
        assert fig.axes

    def test_switch_tool_does_not_replot(self):
        host, ui, fig, container = self._make_host()
        manager = host._manager
        n_axes = len(fig.axes)
        manager.set_tool("RangeStatsTool")
        assert manager.is_active("RangeStatsTool")
        assert len(fig.axes) == n_axes

    def test_teardown_clears_host(self):
        host, ui, fig, container = self._make_host()
        teardown_manager(fig)
        assert host._manager._cids == []
        assert ui.cleared >= 1

    def test_running_unrelated_action_does_not_clear_interactions(self):
        # a non-visual action must leave the live plot/interactions intact
        host, ui, fig, container = self._make_host()
        cleared_before = ui.cleared
        act = _DummyUnrelated(context_key=GCK)
        act.container = container
        act._message = ui.message
        act()
        assert ui.cleared == cleared_before
        assert host._manager.is_active("EventRangesOverlay")


# ---------------------------------------------------------------------------
# Tool-level runtime configuration
# ---------------------------------------------------------------------------


class TestToolConfig:
    def _make_host(self):
        return TestInteractiveHost()._make_host()

    def test_auto_bind_fills_none_only(self):
        from dac.modules.event_log.interactions import EventRangesOverlay

        container = Container()
        coll = EventLogCollection(name="Log")
        coll.add_entry("1.0", "2.5", "Knock")
        container.CurrentContext.add_node(coll)
        fig = _new_figure()
        fig.add_subplot(111)
        ctx = InteractionContext(fig, container)
        overlay = EventRangesOverlay(ctx)

        overlay.auto_bind()
        assert overlay.events == [coll]
        # explicit value survives a second auto_bind
        overlay.events = []
        overlay.auto_bind()
        assert overlay.events == []

    def test_get_construct_config_serialises_names(self):
        host, ui, fig, container = self._make_host()
        cfg = host.get_tool_construct_config("EventRangesOverlay")
        assert cfg["events"] == ["Log"]
        assert cfg["label_axes_index"] == 0

    def test_apply_tool_config_persist_flag(self):
        host, ui, fig, container = self._make_host()
        before = dict(host._construct_config["RangeStatsTool"])

        ok = host.apply_tool_config(
            "RangeStatsTool", {"channels": ["AccelX"], "plot_dt": 0.25}, persist=False
        )
        assert ok
        assert host._construct_config["RangeStatsTool"] == before
        assert host._manager.get("RangeStatsTool").channels[0] is container.get_node_of_type(
            "AccelX", TimeChannel
        )
        assert host._manager.get("RangeStatsTool").plot_dt == 0.25

        ok = host.apply_tool_config(
            "RangeStatsTool", {"channels": ["AccelX"], "plot_dt": 0.5}, persist=True
        )
        assert ok
        assert host._construct_config["RangeStatsTool"]["plot_dt"] == 0.5

    def test_reconfigure_rebuilds_overlay(self):
        from dac.modules.event_log.interactions import EventRangesOverlay

        container = Container()
        coll = EventLogCollection(name="Log")
        coll.add_entry("1.0", "2.5", "Knock")
        container.CurrentContext.add_node(coll)
        fig = _new_figure()
        ax = fig.add_subplot(111)
        ctx = InteractionContext(fig, container)
        overlay = EventRangesOverlay(ctx)
        overlay._active = True
        overlay.attach()
        with_spans = len(ax.get_children())
        assert with_spans > 0

        overlay.apply_construct_config({"events": [], "label_axes_index": 0})
        assert len(ax.get_children()) < with_spans
        overlay.detach()

    def test_reconfigure_redraws(self):
        from dac.modules.event_log.interactions import EventRangesOverlay

        container = Container()
        coll = EventLogCollection(name="Log")
        coll.add_entry("1.0", "2.5", "Knock")
        container.CurrentContext.add_node(coll)
        fig = _new_figure()
        fig.add_subplot(111)
        ctx = InteractionContext(fig, container)
        overlay = EventRangesOverlay(ctx)
        overlay._active = True
        overlay.attach()

        calls = []
        fig.canvas.draw_idle = lambda *a, **k: calls.append(1)
        overlay.apply_construct_config(
            {"events": list(container.CurrentContext.nodes_of_type(EventLogCollection)),
             "label_axes_index": 0}
        )
        assert calls
        overlay.detach()

    def test_activation_refused_when_unavailable(self):
        from dac.modules.event_log.interactions import EventRangesOverlay

        container = Container()  # no EventLogCollection
        fig = _new_figure()
        fig.add_subplot(111)
        ui = _UIStub()
        ctx = InteractionContext(fig, container, ui=ui)
        manager = PlotInteractionManager(ctx)
        manager.add(EventRangesOverlay(ctx))
        manager.set_tool("EventRangesOverlay")
        assert not manager.is_active("EventRangesOverlay")
        assert manager.active_tool is None
        assert ui.messages

    def test_unedited_config_applies(self):
        # an editor round-trip without changes must stay parseable
        host, ui, fig, container = self._make_host()
        cfg = host.get_tool_construct_config("FreqLinesTimeTool")
        assert cfg["stages"] is None
        assert host.apply_tool_config("FreqLinesTimeTool", cfg, persist=False)

    def test_plot_axes_snapshot(self):
        host, ui, fig, container = self._make_host()
        n_plot = len(host._manager.ctx.plot_axes)
        fig.add_axes([0.0, 0.0, 0.1, 0.1])
        assert len(fig.get_axes()) == n_plot + 1
        assert len(host._manager.ctx.plot_axes) == n_plot

    def test_switch_pushes_to_observer(self):
        # the GUI registers an observer that reloads the editor on switch
        host, ui, fig, container = self._make_host()
        events = []
        host._manager.add_observer(lambda name, active: events.append((name, active)))
        host._manager.set_tool("RangeStatsTool")
        assert ("RangeStatsTool", True) in events
        events.clear()
        host._manager.refresh_current()
        assert events == [("RangeStatsTool", True)]


# ---------------------------------------------------------------------------
# Drivetrain frequency-line tools
# ---------------------------------------------------------------------------


class TestDrivetrainTools:
    def _setup(self):
        from dac.modules.drivetrain import GearboxDefinition, GearStage
        from dac.modules.timedata import TimeData

        gb = GearboxDefinition(
            "test",
            stages=[
                GearStage({"Wheel": 50, "Pinion": 25}),
                GearStage({"Wheel": 40, "Pinion": 20}),
            ],
        )
        speed = TimeData(name="speed", y=np.full(100, 3000.0), dt=0.1)
        container = Container()
        container.CurrentContext.add_node(gb)
        container.CurrentContext.add_node(speed)
        fig = _new_figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(100))
        return container, fig, ax

    def test_time_tool_draws_lines(self):
        from dac.modules.drivetrain.interactions import FreqLinesTimeTool

        container, fig, ax = self._setup()
        ctx = InteractionContext(fig, container)
        assert FreqLinesTimeTool.available(ctx)
        tool = FreqLinesTimeTool(ctx)
        tool.attach()
        tool.on_press(_Event(inaxes=ax, xdata=10.0, button=1))
        assert len(tool._lines) > 0
        tool.detach()
        assert tool._lines == []

    def test_spectrum_tool_draws_lines(self):
        from dac.modules.drivetrain.interactions import FreqLinesSpectrumTool

        container, fig, ax = self._setup()
        ctx = InteractionContext(fig, container)
        assert FreqLinesSpectrumTool.available(ctx)
        tool = FreqLinesSpectrumTool(ctx)
        tool.attach()
        assert len(tool._lines) > 0
        tool.detach()
        assert tool._lines == []


# ---------------------------------------------------------------------------
# Event log inspect tool (non-dialog)
# ---------------------------------------------------------------------------


class TestInspectTool:
    def test_range_table(self):
        from dac.modules.event_log.interactions import InspectTool

        container = Container()
        container.CurrentContext.add_node(_channel(name="AccelX", dt=0.1, unit="g"))
        fig = _new_figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(100) * 0.1, np.sin(np.arange(100) * 0.1))

        ui = _UIStub()
        ctx = InteractionContext(fig, container, ui=ui)
        tool = InspectTool(ctx)
        assert InspectTool.available(ctx)
        tool.attach()
        tool.t_start = 1.0
        tool.t_end = 2.0
        tool.on_confirm()
        assert ui.stats is not None
        assert ui.stats["headers"]["row"] == ["AccelX [g]"]
        tool.detach()
