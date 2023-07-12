from dac.core.actions import VAB, PAB, SAB
from . import BallBearing, GearboxDefinition
from dac.modules.timedata import TimeData

class BearingInputStage(int): # cannot use namedtuple(BallBearing, BearingInputStage) because no conversion for namedtuple
    pass

class ShowFreqLinesTime(VAB):
    CAPTION = "Plot frequency lines"
    def __call__(self, bearings: list[tuple[BallBearing, BearingInputStage]]):
        ...

class ShowFreqLinesFreq(VAB):
    CAPTION = "Mark specific frequencies or sidebands"
    def __call__(self, gearbox: GearboxDefinition, speed_channel: TimeData, speed_on_output: bool=True):
        if not speed_channel or not gearbox:
            return
        fig = self.figure
        ax = fig.gca()
        canvas = self.canvas

        n = len(gearbox.stages)