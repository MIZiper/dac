from enum import Enum

class GearStage:
    class StageType(Enum):
        Unknown = 0
        Planetary = 1
        Parallel = 2

class GearboxDefinition:
    def __init__(self, stages):
        ...

class Bearing:
    ...