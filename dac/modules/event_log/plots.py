"""Plot helpers for rendering event log overlays on existing Matplotlib axes."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from . import EventLogCollection, parse_time, time_midpoint

_DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def overlay_events(
    events: list["EventLogCollection"],
    label_axes_index: int = 0,
    color_cycle: list[str] | None = None,
    figure: "Figure | None" = None,
) -> None:
    """Draw event ranges as ``axvspan`` patches on all axes of *figure*.

    Text labels are placed only on the axis at *label_axes_index*
    (default 0 — the first subplot).

    Parameters
    ----------
    events:
        Event log collections whose entries are rendered as coloured
        spans.  Each collection uses its ``color`` attribute when set,
        otherwise the next entry in *color_cycle* is used.
    label_axes_index:
        Index of the Matplotlib axes that receives text labels.
    color_cycle:
        List of colour strings used for auto-assignment.
    figure:
        Target figure; defaults to ``plt.gcf()``.
    """
    if figure is None:
        import matplotlib.pyplot as plt

        figure = plt.gcf()

    all_axes = figure.get_axes()
    if not all_axes:
        return

    label_ax = all_axes[min(label_axes_index, len(all_axes) - 1)]
    colors = color_cycle or _DEFAULT_COLORS

    for i, coll in enumerate(events):
        if not isinstance(coll, EventLogCollection):
            continue

        color = coll.color or colors[i % len(colors)]

        for entry in coll.entries:
            t0 = parse_time(entry.start)
            t1 = parse_time(entry.end)
            if t0 is None or t1 is None:
                continue

            for ax in all_axes:
                ax.axvspan(t0, t1, alpha=0.15, color=color)

            if entry.name:
                mid = time_midpoint(t0, t1)
                _, y_max = label_ax.get_ylim()
                label_ax.text(
                    mid,
                    y_max * 0.95,
                    entry.name,
                    color=color,
                    fontsize=8,
                    ha="center",
                    va="top",
                    rotation=90,
                )
