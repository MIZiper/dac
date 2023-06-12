from enum import Enum
import numpy as np
from dac.core.data import DataBase

class GearStage:
    class StageType(Enum):
        Unknown = 0
        Planetary = 1
        Parallel = 2

class GearboxDefinition(DataBase):
    def __init__(self, name: str = None, uuid: str = None, stages: dict=None) -> None:
        super().__init__(name, uuid)

        self.stages = stages or [{}, {}]

class BallBearing(DataBase):
    def __init__(self, name: str = None, uuid: str = None, N_balls: int=8, D_ball: float=2, D_pitch: float=12, beta: float=15) -> None:
        super().__init__(name, uuid)
        self.N_balls = N_balls
        self.D_ball = D_ball
        self.D_pitch = D_pitch # = (D_IR+D_OR)/2
        self.beta = beta

    def bpfo(self):
        # outer race defect frequency
        # ~ 0.4 * N_balls
        return self.N_balls/2 * (1-self.D_ball/self.D_pitch*np.cos(np.deg2rad(self.beta)))
    
    def bpfi(self):
        # inner race defect frequency
        # ~ 0.6 * N_balls
        return self.N_balls/2 * (1+self.D_ball/self.D_pitch*np.cos(np.deg2rad(self.beta)))
    
    def bsf(self):
        # ball defect frequency
        return self.D_pitch/self.D_ball*(1-(self.D_ball/self.D_pitch*np.cos(np.deg2rad(self.beta)))**2)
    
    def ftf(self):
        # cage defect frequency
        # ~ 0.4
        return 1/2*(1-self.D_ball/self.D_pitch*np.cos(np.deg2rad(self.beta)))