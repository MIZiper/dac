from dac.core.actions import VAB, PAB, SAB
from . import BallBearing, GearboxDefinition

class BearingInputStage(int): # cannot use namedtuple(BallBearing, BearingInputStage) because no conversion for namedtuple
    pass

class ShowFreqLinesTime(VAB):
    CAPTION = "Plot frequency lines"
    def __call__(self, bearings: list[tuple[BallBearing, BearingInputStage]]):
        ...

class ShowFreqLinesFreq(VAB):
    ...