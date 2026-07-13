"""Chart specification for customisable TimeChannel visualisation.

Provides a declarative way to describe multi-subplot, multi-axis charts
with shared axes, twin y-axes, and side-by-side time-window comparisons.
The configuration is a plain dict (YAML) that can be stored in an
action's ``_construct_config``.

A minimal spec::

    {
        "layout": ["speed", "torque"],
        "axes": {
            "speed": {"chs": ["Speed"]},
            "torque": {"chs": ["Torque"], "share_x": "speed"},
        },
    }

Rendering is handled by :func:`render_spec`.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from matplotlib import gridspec
from matplotlib import rcParams
from matplotlib import dates as mdates
from matplotlib.figure import Figure

from . import TimeChannel, TSChannel
from .plots import is_datetime_type, setup_datetime_axis


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class CoaxSpec:
    """Twin x-axis specification for a subplot.

    Parameters
    ----------
    chs : list[str]
        Channel names plotted on this twin axis.
    ylim :
        Y-axis range ``(ymin, ymax)`` or ``None`` for automatic.
    """

    __slots__ = ("chs", "ylim")

    def __init__(self, chs: list[str], ylim=None):
        self.chs = chs
        self.ylim = ylim

    @classmethod
    def from_dict(cls, d: dict) -> "CoaxSpec":
        return cls(
            chs=_ensure_list(d.get("chs", [])),
            ylim=_ensure_none_or_pair(d.get("y")),
        )


class ChannelSpec:
    """Configuration for a single set of axes (one subplot).

    Parameters
    ----------
    chs : list[str]
        Channel names to plot on this axes.
    xlim :
        X-axis range ``(t_start, t_end)`` or ``None`` for automatic.
    ylim :
        Y-axis range ``(ymin, ymax)`` or ``None`` for automatic.
    height : int
        Relative row height (used for ``height_ratios`` in GridSpec).
    grid : bool
        Show grid lines.
    share_x : str | None
        Name of another subplot whose x-axis this subplot shares.
    share_y : str | None
        Name of another subplot whose y-axis this subplot shares.
    coax : CoaxSpec | None
        Twin x-axis configuration.
    xs_override :
        Per-entry xlim used when the ChartSpec ``xs`` list creates
        cloned columns (set internally during expansion, not by users).
    """

    __slots__ = (
        "chs", "xlim", "ylim", "height", "grid",
        "share_x", "share_y", "coax", "xs_override",
    )

    def __init__(
        self,
        chs: list[str],
        xlim=None,
        ylim=None,
        height: int = 10,
        grid: bool = False,
        share_x: str | None = None,
        share_y: str | None = None,
        coax: CoaxSpec | None = None,
    ):
        self.chs = chs
        self.xlim = xlim
        self.ylim = ylim
        self.height = height
        self.grid = grid
        self.share_x = share_x
        self.share_y = share_y
        self.coax = coax
        self.xs_override = None  # set during xs expansion

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelSpec":
        coax = None
        if "coax" in d:
            coax = CoaxSpec.from_dict(d["coax"])
        return cls(
            chs=_ensure_list(d.get("chs", [])),
            xlim=_ensure_none_or_pair(d.get("x")),
            ylim=_ensure_none_or_pair(d.get("y")),
            height=int(d.get("height", 10)),
            grid=_ensure_bool(d.get("grid", False)),
            share_x=_ensure_string_or_none(d.get("share_x")),
            share_y=_ensure_string_or_none(d.get("share_y")),
            coax=coax,
        )

    def clone(self) -> "ChannelSpec":
        c = copy.copy(self)
        c.chs = list(self.chs)
        if self.coax is not None:
            c.coax = CoaxSpec(chs=list(self.coax.chs), ylim=self.coax.ylim)
        return c


class ChartSpec:
    """Complete chart specification.

    Parameters
    ----------
    title : str
        Figure title (set via ``figure.suptitle``).
    layout :
        2-D grid of subplot names.  Each row is a ``list[str]``.
        When *xs* is present and *layout* is a flat ``list[str]``
        the flat list is expanded into columns (one per time window).
    axes :
        Named subplot specifications keyed by the names used in *layout*.
    xs :
        List of ``[t_start, t_end]`` time windows.  When *layout* is
        flat (one column) each window creates a new column showing the
        same channels side by side.  When *layout* is already 2-D,
        *xs* entries map to rows by index, setting *xlim* per row.
    ds_step : float | None
        Downsampling step in seconds.  ``None`` means no downsampling.
    legend : str or None
        ``None``: legend on each axes at upper right.
        ``"single"``: legend only on the first subplot.
        ``"none"``: no legend.
        Any other string is passed to ``ax.legend(loc=...)``.
    """

    __slots__ = ("title", "layout", "axes", "xs", "ds_step", "legend")

    def __init__(
        self,
        title: str = "",
        layout: list[list[str]] | None = None,
        axes: dict[str, ChannelSpec] | None = None,
        xs: list | None = None,
        ds_step: float | None = None,
        legend: str | None = None,
    ):
        self.title = title
        self.layout = layout or []
        self.axes = axes or {}
        self.xs = xs or []
        self.ds_step = ds_step
        self.legend = legend


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def spec_from_dict(d: dict) -> ChartSpec:
    """Build a :class:`ChartSpec` from a plain dictionary (e.g. YAML)."""
    axes_raw: dict = d.get("axes", {})
    axes = {name: ChannelSpec.from_dict(cfg) for name, cfg in axes_raw.items()}

    layout_raw = d.get("layout", [])
    xs = d.get("xs", [])

    spec = ChartSpec(
        title=d.get("title", ""),
        layout=[],
        axes=axes,
        xs=xs,
        ds_step=d.get("ds_step"),
        legend=d.get("legend"),
    )

    # Expand xs columns / row xlims BEFORE normalising to 2-D
    spec.layout, spec.axes = _expand_xs(spec, layout_raw)

    # If no expansion happened, just normalise
    if not spec.layout:
        spec.layout = _normalise_layout(layout_raw)

    return spec


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_spec(
    spec: ChartSpec,
    channels: list[TimeChannel | TSChannel],
    figure: Figure,
) -> None:
    """Render *spec* on *figure* using data from *channels*.

    1. Expand *layout* with *xs* columns (if applicable).
    2. Create ``GridSpec`` subplots with shared-axes references.
    3. Plot each channel, handling twin axes and downsampling.
    4. Apply styling: limits, grid, legend, tick dedup.
    """
    _ch_by_name = {ch.name: ch for ch in channels}
    layout = spec.layout
    axes = spec.axes

    n_rows = len(layout)
    if n_rows == 0:
        return

    # ---- 1. grid dimensions ----
    n_cols = max(len(row) for row in layout)

    height_ratios = []
    for r in range(n_rows):
        first_spec = axes.get(layout[r][0])
        height_ratios.append(first_spec.height if first_spec else 10)

    width_ratios = None
    if spec.xs and len(spec.xs) > 1 and n_cols > 1:
        try:
            # All xs entries should be uniform type so we can sort
            flat = [float(v) for pair in spec.xs for v in pair]
            dur = []
            for i in range(0, len(flat), 2):
                dur.append(abs(flat[i + 1] - flat[i]))
            width_ratios = dur[:n_cols]
        except (ValueError, TypeError):
            pass

    # ---- 2. create GridSpec and subplots ----
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
    )

    mpl_axes: dict[str, Any] = {}       # subplot-name → mpl Axes
    mpl_coax: dict[str, Any] = {}       # subplot-name → twinx mpl Axes

    for r in range(n_rows):
        for c in range(len(layout[r])):
            name = layout[r][c]
            ax_spec = axes.get(name)
            if ax_spec is None:
                continue

            sharex = mpl_axes.get(ax_spec.share_x) if ax_spec.share_x else None
            sharey = mpl_axes.get(ax_spec.share_y) if ax_spec.share_y else None

            mpl_ax = figure.add_subplot(gs[r, c], sharex=sharex, sharey=sharey)
            mpl_axes[name] = mpl_ax

            if ax_spec.coax is not None:
                mpl_coax[name] = mpl_ax.twinx()

    # ---- 3. prepare xs per column (when xs expansion applies) ----
    col_xlim: list | None = None
    if spec.xs and n_cols > 1:
        try:
            col_xlim = [tuple([float(v) for v in pair]) for pair in spec.xs]
            # Pad to n_cols
            col_xlim += [None] * (n_cols - len(col_xlim))
        except (ValueError, TypeError):
            col_xlim = None

    # ---- 4. plot data ----
    color_cycle = rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx = 0

    for r in range(n_rows):
        for c in range(len(layout[r])):
            name = layout[r][c]
            ax_spec = axes.get(name)
            if ax_spec is None or name not in mpl_axes:
                continue

            ax = mpl_axes[name]
            coax_ax = mpl_coax.get(name)

            # Resolve xlim
            xlim = ax_spec.xlim
            if col_xlim is not None and c < len(col_xlim) and col_xlim[c] is not None:
                xlim = col_xlim[c]

            # Collect channel names to plot (primary + coax)
            _plot_channels(
                ax, ax_spec.chs, _ch_by_name, xlim, spec.ds_step,
                color_cycle, color_idx, coax_ax is not None,
            )
            color_idx += len(ax_spec.chs)

            if coax_ax is not None and ax_spec.coax is not None:
                _plot_channels(
                    coax_ax, ax_spec.coax.chs, _ch_by_name, xlim,
                    spec.ds_step, color_cycle, color_idx, False,
                )
                color_idx += len(ax_spec.coax.chs)

            # ---- limits ----
            if xlim is not None:
                ax.set_xlim(xlim)
            if ax_spec.ylim is not None:
                ax.set_ylim(ax_spec.ylim)
            if coax_ax is not None and ax_spec.coax is not None:
                if ax_spec.coax.ylim is not None:
                    coax_ax.set_ylim(ax_spec.coax.ylim)

            # ---- grid ----
            ax.grid(ax_spec.grid)

            # ---- legend ----
            if spec.legend == "none":
                pass
            elif spec.legend == "single":
                if r == 0 and c == 0:
                    _add_legend(ax, coax_ax)
            elif spec.legend is not None:
                ax.legend(loc=spec.legend)
            else:
                _add_legend(ax, coax_ax)

    # ---- 5. datetime formatting ----
    for name, ax in mpl_axes.items():
        for line in ax.get_lines():
            if len(line.get_xdata()) > 0:
                if is_datetime_type(line.get_xdata()):
                    setup_datetime_axis(ax)
                    break

    # ---- 6. tick label dedup for shared axes ----
    _deduplicate_tick_labels(mpl_axes, axes, layout)

    # ---- 7. title ----
    if spec.title:
        figure.suptitle(spec.title)

    figure.tight_layout()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_layout(raw) -> list[list[str]]:
    """Turn a flat ``[str, ...]`` or ``[list[str], ...]`` into 2-D."""
    if not raw:
        return []
    if isinstance(raw[0], list):
        return raw
    return [[name] for name in raw]


def _expand_xs(
    spec: ChartSpec,
    layout_raw: list,
) -> tuple[list[list[str]], dict[str, ChannelSpec]]:
    """Apply *xs* expansion.

    Returns ``(expanded_layout, expanded_axes)``.  Returns ``([], axes)``
    when no expansion applies, signalling the caller to normalise.
    """
    xs = spec.xs
    if not layout_raw or not xs or len(xs) == 0:
        return [], spec.axes

    # Detect flat layout: [str, str, ...]
    is_flat = layout_raw and isinstance(layout_raw[0], str)

    if is_flat and len(xs) > 1:
        new_axes: dict[str, ChannelSpec] = {}
        new_layout: list[list[str]] = []
        for orig_name in layout_raw:
            orig_spec = spec.axes.get(orig_name)
            if orig_spec is None:
                continue
            new_row: list[str] = []
            for j, x_pair in enumerate(xs):
                clone_name = f"{orig_name}_{j}"
                clone = orig_spec.clone()
                clone.xs_override = tuple(x_pair)
                new_axes[clone_name] = clone
                new_row.append(clone_name)
            new_layout.append(new_row)
        return new_layout, new_axes

    # Already 2-D layout: xs maps to rows by index
    if layout_raw and isinstance(layout_raw[0], list):
        new_axes = {name: ax_spec.clone() for name, ax_spec in spec.axes.items()}
        for row_i, x_pair in enumerate(xs):
            if row_i >= len(layout_raw):
                break
            for name in layout_raw[row_i]:
                if name in new_axes:
                    new_axes[name].xs_override = tuple(x_pair)
        return _normalise_layout(layout_raw), new_axes

    return [], spec.axes


def _plot_channels(
    ax,
    ch_names: list[str],
    ch_by_name: dict[str, TimeChannel | TSChannel],
    xlim,
    ds_step: float | None,
    color_cycle: list,
    base_color_idx: int,
    is_coax: bool,
):
    """Plot *ch_names* on *ax*, looking up data from *ch_by_name*."""
    for i, ch_name in enumerate(ch_names):
        ch = ch_by_name.get(ch_name)
        if ch is None:
            continue

        t, y, _dt = ch.get_merged_data(
            t_start=xlim[0] if xlim else None,
            t_end=xlim[1] if xlim else None,
            target_fs=(1.0 / ds_step) if ds_step else None,
        )
        if len(y) == 0:
            continue

        color = color_cycle[(base_color_idx + i) % len(color_cycle)]
        label = f"{ch.name} [{ch.y_unit}]"

        if is_coax:
            ax.plot(t, y, label=label, color=color, linestyle="--")
        else:
            ax.plot(t, y, label=label, color=color)


def _add_legend(ax, coax_ax=None):
    """Add legend(s) for primary and twin axes."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")
    if coax_ax is not None:
        handles2, labels2 = coax_ax.get_legend_handles_labels()
        if handles2:
            coax_ax.legend(loc="upper left")


def _deduplicate_tick_labels(
    mpl_axes: dict[str, Any],
    axes_spec: dict[str, ChannelSpec],
    layout: list[list[str]],
):
    """Hide tick labels on subplots that share axes with others.

    For shared X: hide labels on all but the bottom-most subplot in
    each column.  For shared Y: hide labels on all but the left-most
    subplot in each row.
    """
    if not layout:
        return

    # Build a map: which axes share which mpl Axes
    share_x_groups: dict[int, list[tuple[int, int]]] = {}
    share_y_groups: dict[int, list[tuple[int, int]]] = {}

    mpl_list = {}  # id(mpl_ax) → mpl_ax

    for r, row in enumerate(layout):
        for c, name in enumerate(row):
            mpl_ax = mpl_axes.get(name)
            if mpl_ax is None:
                continue
            mpl_list[id(mpl_ax)] = mpl_ax
            share_x_groups.setdefault(id(mpl_ax), []).append((r, c))
            share_y_groups.setdefault(id(mpl_ax), []).append((r, c))

    # For each share_x group, hide x labels on all but bottom-most
    for ax_id, cells in share_x_groups.items():
        share_refs = set()
        for r, c in cells:
            spec = axes_spec.get(layout[r][c])
            if spec and spec.share_x:
                target = mpl_axes.get(spec.share_x)
                if target is not None:
                    share_refs.add(id(target))
        if not share_refs:
            continue
        for r, c in cells:
            # Keep bottom-most visible
            is_bottom = r == max(row_i for row_i, _ in cells)
            if not is_bottom:
                mpl_axes[layout[r][c]].tick_params(labelbottom=False)

    # For each share_y group, hide y labels on all but left-most
    for ax_id, cells in share_y_groups.items():
        share_refs = set()
        for r, c in cells:
            spec = axes_spec.get(layout[r][c])
            if spec and spec.share_y:
                target = mpl_axes.get(spec.share_y)
                if target is not None:
                    share_refs.add(id(target))
        if not share_refs:
            continue
        for r, c in cells:
            is_leftmost = c == min(col_j for _, col_j in cells)
            if not is_leftmost:
                mpl_axes[layout[r][c]].tick_params(labelleft=False)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _ensure_list(val) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def _ensure_none_or_pair(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return (val[0], val[1])
    return None


def _ensure_string_or_none(val):
    if val is None:
        return None
    return str(val)


def _ensure_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "on", "yes", "1")
    return bool(val)
