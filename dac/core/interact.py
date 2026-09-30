"""Live, switchable interaction layers on top of a rendered figure.

This module decouples *base rendering* (a normal ``VisualizeActionBase``)
from *interaction* (mouse tools and passive overlays that can be toggled
on a live figure without re-plotting the data).

Concepts
--------
``PlotInteraction``
    A lightweight, toggleable overlay or tool bound to a figure.  It has a
    tiny config signature (see ``_SIGNATURE``) so the existing annotation
    resolution in :class:`~dac.core.Container` can inject data nodes.

``OverlayBase`` / ``ToolBase``
    Overlays are passive and stackable (event spans, frequency lines).
    Tools are mutually exclusive and own the mouse (range selection,
    statistics, context creation).

``PlotInteractionManager``
    Owns the interactions of one figure, routes mouse events to the one
    active tool, and handles ``attach`` / ``detach`` so switching never
    re-plots the data.

``InteractivePlotAction``
    A host ``VAB`` that renders a base action once and then installs the
    declared interactions.  Its config is a dict signature keyed by the
    interaction class name, mirroring :class:`~dac.core.actions.SequenceActionBase`.
"""

from __future__ import annotations

import inspect
from typing import Any, Iterable

from matplotlib.figure import Figure

from dac.core import Container, GCK, NodeBase, DataNode
from dac.core.actions import VAB, SequenceActionBase
from dac.core.exceptions import ActionConfigError


__all__ = [
    "InteractionContext",
    "PlotInteraction",
    "OverlayBase",
    "ToolBase",
    "PlotInteractionManager",
    "InteractivePlotAction",
]


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class InteractionContext:
    """Everything an interaction needs to operate on."""

    def __init__(
        self,
        figure: Figure,
        container: Container,
        ui: Any = None,
        base_action: Any = None,
    ) -> None:
        self.figure = figure
        self.container = container
        self.ui = ui
        self.base_action = base_action

    @property
    def axes(self) -> list:
        return self.figure.get_axes()

    @property
    def current_context(self):
        return self.container.CurrentContext if self.container is not None else None

    def find_nodes(self, node_type: type) -> list[DataNode]:
        """All nodes of *node_type* visible from the current context.

        Searches the active context first, then the global context.
        """
        if self.container is None:
            return []
        found: list[DataNode] = []
        seen: set[int] = set()

        for ctx in (self.container.CurrentContext, self.container.contexts.get(GCK)):
            if ctx is None:
                continue
            for node in ctx.nodes_of_type(node_type):
                if id(node) not in seen:
                    seen.add(id(node))
                    found.append(node)
        return found


# ---------------------------------------------------------------------------
# Interaction base classes
# ---------------------------------------------------------------------------


class PlotInteraction:
    """Base class for a toggleable interaction on a figure.

    Subclasses declare their configuration by adding annotated parameters
    with defaults to ``__init__`` (after ``ctx``).  The signature is picked
    up automatically and resolved through
    :meth:`~dac.core.Container.prepare_params_for_action`.
    """

    CAPTION = "Interaction"
    GROUP = "overlay"  # "overlay" (stackable) or "tool" (exclusive)
    DEFAULT_ON = False
    _SIGNATURE = None
    _VALID_PARAM_NAMES: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        sig = inspect.signature(cls.__init__)
        params = [
            p
            for name, p in sig.parameters.items()
            if name not in ("self", "ctx")
        ]
        cls._SIGNATURE = inspect.Signature(params)
        cls._VALID_PARAM_NAMES = frozenset(p.name for p in params)

    def __init__(self, ctx: InteractionContext) -> None:
        self.ctx = ctx
        self._active = False
        self._cids: list[int] = []
        self._artists: list[Any] = []
        self._widgets: list[Any] = []

    # -- capability --------------------------------------------------------

    @classmethod
    def available(cls, ctx: InteractionContext) -> bool:
        """Whether this interaction makes sense for the current context."""
        return True

    # -- lifecycle (override) ---------------------------------------------

    def attach(self) -> None:
        """Create artists / connect events. Called when activated."""

    def detach(self) -> None:
        """Remove artists and disconnect events. Called when deactivated.

        Subclasses overriding this must call ``super().detach()`` so the
        tracked artists / cids are cleaned up.
        """
        self._cleanup()

    # -- helpers -----------------------------------------------------------

    def connect(self, event_name: str, func) -> int:
        """Connect a matplotlib event and track the cid for cleanup."""
        cid = self.ctx.figure.canvas.mpl_connect(event_name, func)
        self._cids.append(cid)
        return cid

    def track(self, artist):
        """Track an artist (or anything with ``.remove()``) for cleanup."""
        self._artists.append(artist)
        return artist

    def _disconnect(self) -> None:
        canvas = self.ctx.figure.canvas
        for cid in self._cids:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._cids.clear()

    def _remove_artists(self) -> None:
        for artist in self._artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._artists.clear()

    def _remove_widgets(self) -> None:
        for widget in self._widgets:
            try:
                widget.disconnect_events()
            except Exception:
                pass
        self._widgets.clear()

    def _cleanup(self) -> None:
        self._disconnect()
        self._remove_artists()
        self._remove_widgets()

    def redraw(self) -> None:
        self.ctx.figure.canvas.draw_idle()

    # -- routed mouse handlers (used by ToolBase subclasses) --------------

    def on_press(self, event) -> None: ...
    def on_release(self, event) -> None: ...
    def on_motion(self, event) -> None: ...


class OverlayBase(PlotInteraction):
    """Passive, stackable interaction (spans, lines, annotations)."""

    GROUP = "overlay"


class ToolBase(PlotInteraction):
    """Active, mutually-exclusive interaction that owns the mouse."""

    GROUP = "tool"


# ---------------------------------------------------------------------------
# Reusable range-selection tool
# ---------------------------------------------------------------------------


def _axes_have_datetime(axes) -> bool:
    import numpy as np

    for ax in axes:
        for line in ax.get_lines():
            xdata = line.get_xdata()
            if len(xdata) and np.asarray(xdata).dtype.kind == "M":
                return True
    return False


def _num_to_time(x):
    import numpy as np
    import matplotlib.dates as mdates

    dt = mdates.num2date(x).replace(tzinfo=None)
    return np.datetime64(dt.isoformat())


class RangeSelectTool(ToolBase):
    """Drag (range) or click (point) selection over the figure axes.

    Subclasses override :meth:`on_confirm` to act on the selection (the
    selected span is available as :attr:`t_start` / :attr:`t_end` and is
    converted to ``np.datetime64`` when the axes render dates).
    """

    CAPTION = "Select range"
    _HINT = "Drag to select a range  |  click for a point"

    def __init__(self, ctx: InteractionContext) -> None:
        super().__init__(ctx)
        self.t_start = None
        self.t_end = None
        self._press_x = None
        self._dragging = False
        self._span_artists: list = []
        self._vline_artists: list = []
        self._axes: list = []
        self._is_datetime = False

    # -- lifecycle ---------------------------------------------------------

    def attach(self) -> None:
        self._axes = list(self.ctx.axes)
        self._is_datetime = _axes_have_datetime(self._axes)
        self.t_start = None
        self.t_end = None
        self._press_x = None
        self._dragging = False
        self._span_artists = []
        self._vline_artists = []
        if self.ctx.ui is not None and hasattr(self.ctx.ui, "message"):
            self.ctx.ui.message(self._HINT, log=False)

    def detach(self) -> None:
        self._clear_selection_artists()
        super().detach()

    # -- helpers -----------------------------------------------------------

    def _normalize(self, x):
        if x is None:
            return None
        if self._is_datetime:
            return _num_to_time(x)
        return x

    def _clear_selection_artists(self) -> None:
        for artist in self._span_artists + self._vline_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._span_artists.clear()
        self._vline_artists.clear()

    def _clear_spans(self) -> None:
        for artist in self._span_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._span_artists.clear()

    # -- mouse -------------------------------------------------------------

    def on_press(self, event) -> None:
        if self.ctx.figure.canvas.widgetlock.locked():
            return
        if event.inaxes not in self._axes:
            return
        if event.button == 3:
            self.on_confirm()
            return
        if event.button != 1 or event.xdata is None:
            return

        self._press_x = event.xdata
        self.t_start = self._normalize(event.xdata)
        self.t_end = self.t_start
        self._dragging = True
        self._clear_selection_artists()
        for ax in self._axes:
            self._span_artists.append(
                ax.axvspan(event.xdata, event.xdata, alpha=0.2, color="green")
            )
        self.redraw()

    def on_motion(self, event) -> None:
        if not self._dragging or event.xdata is None:
            return
        self._clear_spans()
        x0 = min(self._press_x, event.xdata)
        x1 = max(self._press_x, event.xdata)
        for ax in self._axes:
            self._span_artists.append(ax.axvspan(x0, x1, alpha=0.2, color="green"))
        self.redraw()

    def on_release(self, event) -> None:
        if not self._dragging or event.button != 1:
            return
        self._dragging = False
        x = event.xdata
        if x is None or self._press_x is None:
            return

        click_threshold = 0.0
        if self._axes:
            xlim = self._axes[0].get_xlim()
            click_threshold = (xlim[1] - xlim[0]) * 0.005

        self._clear_spans()
        if abs(x - self._press_x) < click_threshold:
            self.t_end = self.t_start
            for ax in self._axes:
                self._vline_artists.append(
                    ax.axvline(self._press_x, color="red", linestyle="--")
                )
        else:
            t0, t1 = sorted([self._normalize(self._press_x), self._normalize(x)])
            self.t_start, self.t_end = t0, t1
            for ax in self._axes:
                self._span_artists.append(
                    ax.axvspan(self._press_x, x, alpha=0.15, color="green")
                )
        self.redraw()
        self._announce_selection()

    def _announce_selection(self) -> None:
        if self.ctx.ui is not None and hasattr(self.ctx.ui, "message"):
            self.ctx.ui.message(f"Selected: {self.t_start} ~ {self.t_end}", log=False)

    def on_confirm(self) -> None:
        """Hook: right-click after a selection exists."""


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class PlotInteractionManager:
    """Owns the interactions of one figure and routes mouse events."""

    def __init__(self, ctx: InteractionContext) -> None:
        self.ctx = ctx
        self._interactions: dict[str, PlotInteraction] = {}
        self._order: list[str] = []
        self._active_tool: str | None = None
        self._observers: list = []

        canvas = ctx.figure.canvas
        self._cids = [
            canvas.mpl_connect("button_press_event", self._dispatch_press),
            canvas.mpl_connect("button_release_event", self._dispatch_release),
            canvas.mpl_connect("motion_notify_event", self._dispatch_motion),
        ]

    # -- registration ------------------------------------------------------

    def add(self, interaction: PlotInteraction, active: bool = False) -> str:
        name = type(interaction).__name__
        self._interactions[name] = interaction
        self._order.append(name)
        if active:
            if interaction.GROUP == "tool":
                self.set_tool(name)
            else:
                self.toggle(name, True)
        return name

    def get(self, name: str) -> PlotInteraction | None:
        return self._interactions.get(name)

    @property
    def interactions(self) -> list[PlotInteraction]:
        return [self._interactions[n] for n in self._order]

    # -- toggling ----------------------------------------------------------

    def is_active(self, name: str) -> bool:
        it = self._interactions.get(name)
        return bool(it and it._active)

    def toggle(self, name: str, on: bool | None = None) -> None:
        it = self._interactions.get(name)
        if it is None:
            return
        if on is None:
            on = not it._active
        if on and not it._active:
            it.attach()
            it._active = True
            self._notify(name, True)
        elif not on and it._active:
            it.detach()
            it._active = False
            self._notify(name, False)
        it.redraw()

    def set_tool(self, name: str | None) -> None:
        if self._active_tool == name:
            return
        if self._active_tool is not None:
            self.toggle(self._active_tool, False)
        self._active_tool = name
        if name is not None:
            self.toggle(name, True)

    @property
    def active_tool(self) -> str | None:
        return self._active_tool

    # -- observers (for toolbar sync) -------------------------------------

    def add_observer(self, fn) -> None:
        self._observers.append(fn)

    def _notify(self, name: str, active: bool) -> None:
        for fn in self._observers:
            try:
                fn(name, active)
            except Exception:
                pass

    # -- teardown ----------------------------------------------------------

    def clear(self) -> None:
        for name in list(self._order):
            self.toggle(name, False)
        canvas = self.ctx.figure.canvas
        for cid in self._cids:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._cids.clear()
        self._interactions.clear()
        self._order.clear()
        self._active_tool = None

    # -- event dispatch ----------------------------------------------------

    def _active_tool_interaction(self) -> PlotInteraction | None:
        if self._active_tool is None:
            return None
        return self._interactions.get(self._active_tool)

    def _dispatch_press(self, event) -> None:
        tool = self._active_tool_interaction()
        if tool is not None:
            tool.on_press(event)

    def _dispatch_release(self, event) -> None:
        tool = self._active_tool_interaction()
        if tool is not None:
            tool.on_release(event)

    def _dispatch_motion(self, event) -> None:
        tool = self._active_tool_interaction()
        if tool is not None:
            tool.on_motion(event)


# ---------------------------------------------------------------------------
# Host action
# ---------------------------------------------------------------------------


class InteractivePlotAction(VAB):
    """Render a base plot once and install switchable interactions.

    Subclass with ``base`` (a ``VisualizeActionBase`` class) and
    ``interactions`` (a sequence of ``PlotInteraction`` classes)::

        class InteractiveTimePlotAction(
            InteractivePlotAction,
            base=SpecPlotAction,
            interactions=[EventRangesOverlay, FreqLinesOverlay, RangeStatsTool],
        ):
            CAPTION = "Interactive time plot"

    The config signature is a dict keyed by class name (like SAB), so the
    action editor and annotation resolution work unchanged.
    """

    CAPTION = "Interactive plot"
    _BASE = None
    _INTERACTIONS: tuple[type[PlotInteraction], ...] = ()

    def __init_subclass__(cls, base=None, interactions=None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if base is not None:
            cls._BASE = base
        if interactions is not None:
            cls._INTERACTIONS = tuple(interactions)

        if cls._BASE is not None:
            sig: dict[str, inspect.Signature] = {
                cls._BASE.__name__: cls._BASE._SIGNATURE
            }
            for inter in cls._INTERACTIONS:
                sig[inter.__name__] = inter._SIGNATURE
            cls._SIGNATURE = sig
            cls._VALID_PARAM_NAMES = frozenset(sig.keys())
            # expose a SAB-like sequence for the generic quick-action helpers
            cls._SEQUENCE = (cls._BASE,) + cls._INTERACTIONS

    def __init__(self, context_key, name: str = None, uuid: str = None) -> None:
        super().__init__(context_key, name, uuid)
        self._construct_config = SequenceActionBase._GetCCFromS(self._SIGNATURE)
        self.ui_host = None
        self._manager: PlotInteractionManager | None = None

    # -- data preparation (worker thread) ---------------------------------

    def prepare(self, **params):
        if self._BASE is None:
            return None

        base_name = self._BASE.__name__
        base = self._BASE(self.context_key)
        if isinstance(base, VAB):
            base._figure = self.figure
        base._progress = self._progress
        base._message = self._message
        base._cancel_check = self._cancel_check

        base_params = params.get(base_name, {}) or {}
        base_prepared = base.prepare(**base_params)
        return {
            "base": base,
            "base_params": base_params,
            "base_prepared": base_prepared,
        }

    # -- render + install (main thread) -----------------------------------

    def __call__(self, **params):
        if self._BASE is None:
            return

        prep = self._prepared
        if prep is None:
            prep = self.prepare(**params)

        base = prep["base"]
        base._figure = self.figure
        base._prepared = prep["base_prepared"]
        base(**prep["base_params"])

        ctx = InteractionContext(
            figure=self.figure,
            container=self.container,
            ui=self.ui_host,
            base_action=base,
        )
        manager = PlotInteractionManager(ctx)

        entries: list[tuple[str, str, str, bool, bool]] = []
        for inter_cls in self._INTERACTIONS:
            name = inter_cls.__name__
            raw = params.get(name) or {}
            try:
                resolved = self.container.prepare_params_for_action(
                    inter_cls._SIGNATURE, raw
                )
            except ActionConfigError:
                resolved = {}
            except Exception as e:
                self._message(f"Interaction {name}: parameter error: {e}")
                resolved = {}

            try:
                inter = inter_cls(ctx, **resolved)
            except Exception as e:
                self._message(f"Interaction {name} failed to initialise: {e}")
                continue

            available = inter_cls.available(ctx)
            default_on = bool(getattr(inter_cls, "DEFAULT_ON", False))
            manager.add(inter, active=(available and default_on))
            entries.append(
                (name, inter_cls.CAPTION, inter_cls.GROUP, default_on and available, available)
            )

        self._manager = manager
        # keep a handle on the canvas so the figure setter can tear it down
        self.figure.canvas._dac_manager = manager

        if self.ui_host is not None and hasattr(self.ui_host, "install_interactions"):
            self.ui_host.install_interactions(manager, entries)
