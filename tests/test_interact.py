"""Tests for the live interaction layer (`dac.core.interact`)."""

import numpy as np
import pytest

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dac.core import Container, GCK
from dac.core.actions import VAB
from dac.core.interact import (
    InteractionContext,
    InteractivePlotAction,
    OverlayBase,
    PlotInteractionManager,
    RangeSelectTool,
    ToolBase,
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
    def __init__(self):
        self.stats = None
        self.messages = []
        self.entries = None

    def message(self, msg, log=True):
        self.messages.append(msg)

    def show_stats(self, stats):
        self.stats = stats

    def install_interactions(self, manager, entries):
        self.entries = entries


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
    def _make(self):
        container = Container()
        fig = _new_figure()
        fig.add_subplot(111)
        ctx = InteractionContext(fig, container)
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

    def test_clear_removes_everything(self):
        manager, _ = self._make()
        manager.clear()
        assert manager.interactions == []
        assert manager._cids == []


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
    def test_render_and_install(self):
        from dac.modules.interactive import InteractiveTimePlotAction

        container = Container()
        container.CurrentContext.add_node(_channel(name="AccelX", dt=0.1, unit="g"))
        coll = EventLogCollection(name="Log")
        coll.add_entry("1.0", "2.5", "Knock")
        container.CurrentContext.add_node(coll)

        ui = _UIStub()
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

        manager = host._manager
        names = [type(i).__name__ for i in manager.interactions]
        assert "EventRangesOverlay" in names
        assert "RangeStatsTool" in names
        assert manager.is_active("EventRangesOverlay")
        assert ui.entries is not None
        assert fig.axes

        n_axes = len(fig.axes)
        manager.set_tool("RangeStatsTool")
        assert len(fig.axes) == n_axes

        manager.clear()
