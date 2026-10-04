"""Host actions that combine a base render with switchable interactions.

Each host renders its base plot once and then exposes the registered
interactions as toggleable overlays/tools (see
:class:`~dac.core.interact.InteractivePlotAction`).  New interactions are
added to the relevant host's ``interactions`` list as they are migrated
from the standalone actions.

Anything that needs a dialog (adding an event log entry, creating an
analysis context) is reported as unavailable on hosts without dialog
support, e.g. the web front-end.
"""

from dac.core.interact import InteractivePlotAction
from dac.modules.pch.actions import SpecPlotAction
from dac.modules.pch.interactions import RangeStatsTool, SelectContextTool
from dac.modules.nvh.actions import ViewFreqDomainAction
from dac.modules.event_log.interactions import (
    AddEventLogTool,
    EventRangesOverlay,
    InspectTool,
)
from dac.modules.drivetrain.interactions import (
    FreqLinesSpectrumTool,
    FreqLinesTimeTool,
)


class InteractiveTimePlotAction(
    InteractivePlotAction,
    base=SpecPlotAction,
    interactions=[
        EventRangesOverlay,
        FreqLinesTimeTool,
        RangeStatsTool,
        SelectContextTool,
        AddEventLogTool,
        InspectTool,
    ],
):
    """Spec plot with switchable event-log / frequency / statistics tools.

    Overlays (event ranges) stack; tools (frequency lines, range
    statistics, context creation, add event, inspect) are mutually
    exclusive and own the mouse.
    """

    CAPTION = "Interactive time plot"


class InteractiveSpectrumPlotAction(
    InteractivePlotAction,
    base=ViewFreqDomainAction,
    interactions=[
        FreqLinesSpectrumTool,
    ],
):
    """FFT spectrum with switchable characteristic-frequency lines."""

    CAPTION = "Interactive spectrum plot"
