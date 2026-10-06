"""Live, switchable interaction layers on top of a rendered figure.

This module decouples *base rendering* (a normal
:class:`~dac.core.actions.VisualizeActionBase`) from *interaction* (mouse
tools and passive overlays) so a rendered figure can switch tools /
overlays **without re-plotting the data**.

Concepts
--------
``PlotInteraction``
    A lightweight, toggleable overlay or tool bound to a figure.  Its tiny
    config signature is derived from ``__init__`` (parameters after
    ``ctx``), so the existing annotation resolution in
    :class:`~dac.core.Container` can inject data nodes.

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
    interaction class name, mirroring
    :class:`~dac.core.actions.SequenceActionBase`.

``InteractionHost``
    Duck-typed protocol implemented by the front-end (PyQt ``MainWindow``
    or the web host).  The core never imports PyQt; dialog-driven
    interactions declare ``REQUIRES_DIALOG = True`` and are reported as
    unavailable when ``host.supports_dialogs`` is false.
"""

from __future__ import annotations

import inspect
import weakref
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
from matplotlib.figure import Figure

from dac.core import Container, GCK, DataNode
from dac.core.actions import VAB, SequenceActionBase


__all__ = [
    "InteractionHost",
    "NullInteractionHost",
    "InteractionEntry",
    "InteractionContext",
    "PlotInteraction",
    "OverlayBase",
    "ToolBase",
    "RangeSelectTool",
    "PlotInteractionManager",
    "InteractivePlotAction",
    "install_manager",
    "teardown_manager",
]


# ---------------------------------------------------------------------------
# Host protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class InteractionHost(Protocol):
    """Minimal interface the front-end provides to interactions.

    The core only uses these methods, so a web host (no Qt) can implement
    the same protocol and reuse every interaction unchanged.
    """

    supports_dialogs: bool

    def message(self, text: str, log: bool = True) -> None: ...

    def show_stats(self, stats: dict) -> None: ...

    def install_interactions(self, manager: "PlotInteractionManager", entries: list) -> None: ...

    def clear_interactions(self) -> None: ...


class NullInteractionHost:
    """No-op host used when no UI is attached (pure headless runs)."""

    supports_dialogs = False

    def message(self, text: str, log: bool = True) -> None:
        pass

    def show_stats(self, stats: dict) -> None:
        pass

    def install_interactions(self, manager: "PlotInteractionManager", entries: list) -> None:
        pass

    def clear_interactions(self) -> None:
        pass


def _supports_dialogs(ui) -> bool:
    return bool(ui is not None and getattr(ui, "supports_dialogs", False))


@dataclass
class InteractionEntry:
    """Declarative description of one interaction, handed to the host."""

    name: str
    caption: str
    group: str
    active: bool
    available: bool
    requires_dialog: bool = False


# ---------------------------------------------------------------------------
# Manager registry (per canvas)
# ---------------------------------------------------------------------------

_managers: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def install_manager(figure: Figure, manager: "PlotInteractionManager") -> None:
    """Bind *manager* to *figure*'s canvas for later teardown."""
    canvas = getattr(figure, "canvas", None)
    if canvas is None:
        return
    try:
        _managers[canvas] = manager
    except TypeError:
        pass


def teardown_manager(figure: Figure) -> None:
    """Detach and clear the interaction manager bound to *figure* (if any)."""
    canvas = getattr(figure, "canvas", None)
    if canvas is None:
        return
    manager = _managers.pop(canvas, None)
    if manager is not None:
        try:
            manager.clear()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class InteractionContext:
    """Everything an interaction needs to operate on."""

    def __init__(
        self,
        figure: Figure,
        container: Container = None,
        ui: Any = None,
        base_action: Any = None,
        host_action: Any = None,
        plot_axes: list = None,
    ) -> None:
        self.figure = figure
        self.container = container
        self.ui = ui
        self.base_action = base_action
        self.host_action = host_action
        # axes that belong to the base plot (captured before interaction
        # panels/widgets are added); tools should select over these only
        self.plot_axes = plot_axes

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
    REQUIRES_DIALOG = False

    _SIGNATURE = None
    _VALID_PARAM_NAMES: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        sig = inspect.signature(cls.__init__)
        params = [
            p for name, p in sig.parameters.items() if name not in ("self", "ctx")
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
        """Whether this interaction makes sense for the current context.

        Dialog-driven interactions are unavailable when the host cannot
        show dialogs (e.g. the web front-end).
        """
        if cls.REQUIRES_DIALOG and not _supports_dialogs(ctx.ui):
            return False
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

    def reconfigure(self) -> None:
        """Rebuild the interaction in place (detach + attach + redraw) after
        a parameter change, without touching the base plot."""
        if not self._active:
            return
        self.detach()
        self.attach()
        self.redraw()

    def is_ready(self) -> bool:
        """Instance-level readiness check run at activation time.

        Subclasses override this to require specific parameters (e.g. a
        selected node).  Parameters are configured explicitly through the
        editor; nothing is auto-filled.  Default: always ready.
        """
        return True

    # -- runtime parameters (tool-level editing) --------------------------

    def _config_value(self, value):
        """Serialise a runtime value to a YAML/basic-type representation."""
        if isinstance(value, DataNode):
            container = self.ctx.container
            if container is not None:
                return container.CurrentContext.get_qualified_name(value)
            return value.name
        if isinstance(value, (list, tuple)):
            return [self._config_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._config_value(v) for k, v in value.items()}
        return value

    def get_construct_config(self) -> dict:
        """Current parameter values as a YAML-friendly dict.

        DataNodes become qualified names.  Still-unset parameters are
        emitted as their default (usually ``None``) rather than an
        annotation hint, so an unedited apply stays parseable.
        """
        cfg: dict = {}
        for name, param in self._SIGNATURE.parameters.items():
            value = getattr(self, name, None)
            if value is None:
                cfg[name] = param.default if param.default is not inspect._empty else None
            else:
                cfg[name] = self._config_value(value)
        return cfg

    def apply_construct_config(self, resolved: dict, reconfigure: bool = True) -> None:
        """Apply already-resolved parameter values to this live instance.

        *resolved* maps parameter names to Python values (DataNodes, not
        names).  Unknown keys are ignored.
        """
        for name, value in (resolved or {}).items():
            if name in self._VALID_PARAM_NAMES:
                setattr(self, name, value)
        if reconfigure:
            self.reconfigure()

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

    def notify(self, text: str) -> None:
        ui = self.ctx.ui
        if ui is not None and hasattr(ui, "message"):
            try:
                ui.message(text, log=False)
            except TypeError:
                ui.message(text)

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
    for ax in axes:
        for line in ax.get_lines():
            xdata = line.get_xdata()
            if len(xdata) and np.asarray(xdata).dtype.kind == "M":
                return True
    return False


def _num_to_time(x) -> np.datetime64:
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
    _HINT = "Drag to select a range  |  click for a point  |  right-click to confirm"

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
        self._axes = list(self.ctx.plot_axes or self.ctx.axes)
        self._is_datetime = _axes_have_datetime(self._axes)
        self.t_start = None
        self.t_end = None
        self._press_x = None
        self._dragging = False
        self._span_artists = []
        self._vline_artists = []
        self.notify(self._HINT)

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
        if self.t_start is not None:
            self.notify(f"Selected: {self.t_start} ~ {self.t_end}")

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
        self._current: str | None = None
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

    def get(self, name: str) -> Optional[PlotInteraction]:
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
            # availability/readiness are evaluated at activation time, so
            # data added after the plot was rendered is picked up
            try:
                available = bool(type(it).available(self.ctx))
            except Exception:
                available = False
            try:
                ready = bool(it.is_ready())
            except Exception:
                ready = False
            if not available:
                self._message(
                    f"{type(it).CAPTION} is not available "
                    "(missing required data or unsupported)."
                )
                return
            if not ready:
                self._message(
                    f"{type(it).CAPTION} needs configuration "
                    "(open it in the editor and set its parameters)."
                )
                return
            it.attach()
            it._active = True
            self._current = name
            self._notify(name, True)
        elif not on and it._active:
            it.detach()
            it._active = False
            self._notify(name, False)
        it.redraw()

    def _message(self, text: str) -> None:
        ui = self.ctx.ui
        if ui is not None and hasattr(ui, "message"):
            try:
                ui.message(text, log=False)
            except TypeError:
                ui.message(text)

    def set_tool(self, name: str | None) -> None:
        if self._active_tool == name:
            return
        if self._active_tool is not None:
            self.toggle(self._active_tool, False)
            self._active_tool = None
        if name is not None:
            self.toggle(name, True)
            # only record it if activation actually succeeded
            if self.is_active(name):
                self._active_tool = name

    @property
    def active_tool(self) -> str | None:
        return self._active_tool

    @property
    def current_name(self) -> str | None:
        """Name of the last activated interaction (tool or overlay)."""
        return self._current

    @property
    def current_interaction(self) -> Optional[PlotInteraction]:
        if self._current is None:
            return None
        return self._interactions.get(self._current)

    def refresh_current(self) -> None:
        """Re-notify observers for the current interaction (editor reload)."""
        if self._current is not None and self.is_active(self._current):
            self._notify(self._current, True)

    def apply_config(self, name: str, resolved: dict, reconfigure: bool = True) -> None:
        """Apply resolved parameters to a registered interaction."""
        it = self._interactions.get(name)
        if it is None:
            return
        it.apply_construct_config(resolved, reconfigure=reconfigure)

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
        self._current = None
        ui = self.ctx.ui
        if ui is not None and hasattr(ui, "clear_interactions"):
            try:
                ui.clear_interactions()
            except Exception:
                pass

    # -- event dispatch ----------------------------------------------------

    def _active_tool_interaction(self) -> Optional[PlotInteraction]:
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
            interactions=[EventRangesOverlay, FreqLinesTimeTool, RangeStatsTool],
        ):
            CAPTION = "Interactive time plot"

    The config signature is a dict keyed by class name (like SAB), so the
    action editor and annotation resolution work unchanged.  ``ui_host``
    is injected by the front-end runner; when it is ``None`` the action
    still renders and attaches interactions in a headless fashion.
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
        self._manager: Optional[PlotInteractionManager] = None

    # -- data preparation (worker thread) ---------------------------------

    def prepare(self, **params):
        if self._BASE is None:
            return None

        base = self._BASE(self.context_key)
        if isinstance(base, VAB):
            base._figure = self.figure
        base._progress = self._progress
        base._message = self._message
        base._cancel_check = self._cancel_check

        base_params = params.get(self._BASE.__name__, {}) or {}
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

        # a fresh render replaces whatever interaction manager was attached
        figure = self.figure
        teardown_manager(figure)

        ui = self.ui_host
        ctx = InteractionContext(
            figure=figure,
            container=self.container,
            ui=ui,
            base_action=base,
            host_action=self,
            plot_axes=list(figure.get_axes()),
        )
        manager = PlotInteractionManager(ctx)

        entries: list[InteractionEntry] = []
        for inter_cls in self._INTERACTIONS:
            name = inter_cls.__name__
            resolved = params.get(name) or {}
            try:
                inter = inter_cls(ctx, **resolved)
            except Exception as e:
                self._message(f"Interaction {name} failed to initialise: {e}")
                continue

            try:
                available = bool(inter_cls.available(ctx))
            except Exception as e:
                self._message(f"Interaction {name} availability check failed: {e}")
                available = False

            default_on = bool(getattr(inter_cls, "DEFAULT_ON", False))
            active = available and default_on
            manager.add(inter, active=active)
            entries.append(
                InteractionEntry(
                    name=name,
                    caption=inter_cls.CAPTION,
                    group=inter_cls.GROUP,
                    active=active,
                    available=available,
                    requires_dialog=bool(
                        getattr(inter_cls, "REQUIRES_DIALOG", False)
                    ),
                )
            )

        self._manager = manager
        install_manager(figure, manager)

        if ui is not None and hasattr(ui, "install_interactions"):
            ui.install_interactions(manager, entries)

    # -- tool-level configuration -----------------------------------------

    def interaction_class(self, name: str) -> Optional[type[PlotInteraction]]:
        for inter_cls in self._INTERACTIONS:
            if inter_cls.__name__ == name:
                return inter_cls
        return None

    def get_tool_construct_config(self, name: str) -> dict:
        """Current YAML-friendly config of interaction *name*.

        Prefers the live instance (so temporary changes are visible when
        the user re-opens the tool); falls back to the persisted action
        config merged over the parameter defaults.
        """
        inter = self._manager.get(name) if self._manager is not None else None
        if inter is not None:
            return inter.get_construct_config()

        inter_cls = self.interaction_class(name)
        if inter_cls is None:
            return {}
        cfg = SequenceActionBase._GetCCFromS(inter_cls._SIGNATURE)
        raw = self._construct_config.get(name)
        if isinstance(raw, dict):
            cfg.update(raw)
        return cfg

    def apply_tool_config(self, name: str, raw_cfg: dict, persist: bool = False) -> bool:
        """Apply an interaction's YAML config to the live instance.

        ``persist=False`` (apply) only updates the running interaction;
        ``persist=True`` (apply+run) also writes the raw config back into
        the action config, so it survives re-runs / project save.
        Returns ``True`` on success.
        """
        inter = self._manager.get(name) if self._manager is not None else None
        if inter is None:
            self._message(f"Interaction '{name}' is not available.")
            return False
        if self.container is None:
            self._message(f"Cannot resolve '{name}' without a container.")
            return False

        try:
            resolved = self.container.prepare_params_for_action(
                type(inter)._SIGNATURE, raw_cfg or {}
            )
        except Exception as e:
            self._message(f"Invalid config for '{name}': {e}")
            return False

        try:
            inter.apply_construct_config(resolved, reconfigure=True)
        except Exception as e:
            self._message(f"Failed to apply '{name}': {e}")
            return False

        if persist:
            self._construct_config[name] = dict(raw_cfg or {})
        return True
