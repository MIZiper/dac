from enum import Enum
import numpy as np
from dac.core.data import DataBase
from collections import defaultdict

class GearStage:
    class StageType(Enum):
        Unknown = 0
        Planetary = 1
        Parallel = 2
    
    def __init__(self, config: dict):
        self.config = config
        match config:
            case {"RG": rg, "PG": pg, "SU": su, "NoP": nop}:
                ratio = 1 + rg/su
                t = GearStage.StageType.Planetary
            case {"Wheel": wheel, "Pinion": pinion}:
                ratio = wheel / pinion
                t = GearStage.StageType.Parallel
            case _:
                ratio = 1
                t = GearStage.StageType.Unknown

        self.ratio = ratio
        self.stage_type = t

    def f(self, input_speed): # order=1
        return input_speed / 60

    def fz(self, input_speed, order=1):
        if self.stage_type==GearStage.StageType.Planetary:
            n = self.config['NoP']
            z = self.config['RG']*order
            Z = round(z/n) * n # in non-factorizing case, find closest (dominant) side band; for factorizing, same as `z`
            _Z = (-1)**(Z>z)*n*(Z!=z)+Z # the other side band
            # _Z = np.sign(z-Z)*n+Z # same

            # TODO: distinguish fz and fzN/fzn, otherwise 2*{fz} is not {2fz}
            
            return input_speed / 60 * Z
        
        elif self.stage_type==GearStage.StageType.Parallel:
            return input_speed / 60 * self.config['Wheel']

    def input_speed(self, output_speed):
        return output_speed / self.ratio

    def get_freq_at_order(self, order: str):
        ...

    def get_order_name_at_freq(self, freq: float, max_order: int, tolerance: int):
        # return the name
        ...

    def get_freqs_labels_at(self, speed: float, speed_on_output: bool=True):
        rst = [
            (
                self.f( self.input_speed(speed) if speed_on_output else speed ),
                "f"
            ),
            (
                self.fz( self.input_speed(speed) if speed_on_output else speed ),
                "fz"
            ),
        ]
        if self.stage_type==GearStage.StageType.Planetary:
            NoP = self.config['NoP']
            rst.append(
                (
                    NoP * self.f( self.input_speed(speed) if speed_on_output else speed ),
                    f"{NoP}pf",
                )
            )
            rst.append(
                (
                    self.f( self.input_speed(speed) if speed_on_output else speed ) * self.config['RG'] / self.config['PG'], # PG rotation speed related to PC
                    f"fw",
                )
            )

        return rst

class GearboxDefinition(DataBase):
    def __init__(self, name: str = None, uuid: str = None, stages: list[GearStage]=None, bearings: list[tuple["BearingInputStage", "BallBearing"]]=None) -> None:
        super().__init__(name, uuid)

        self.stages = stages or []
        self.bearings = bearings or []

    def get_construct_config(self) -> dict:
        if self.stages:
            stgs = [stg.config for stg in self.stages]
        else:
            stgs = [
                {"RG": "<int>", "PG": "<int>", "SU": "<int>", "NoP": "[num of planets]"},
                {"Wheel": "<int>", "Pinion": "<int>"},
            ]

        return {
            "name": self.name,
            "stages": stgs
        }
    
    def apply_construct_config(self, construct_config: dict):
        self.name = construct_config["name"]
        self.stages.clear()
        for stage in construct_config["stages"]:
            self.stages.append(GearStage(stage))

    def get_freqs_labels_at(self, speed: float, speed_on_output: bool=True, choice_bits: int=-1):
        rst = []
        bearing_dict = defaultdict(list[BallBearing])
        for stage, bearing in self.bearings:
            bearing_dict[stage].append(bearing)

        if not speed_on_output:
            for i, stg in enumerate(self.stages):
                if (choice_bits==-1) or (choice_bits & (1<<i)):
                    for freq, lbl in stg.get_freqs_labels_at(speed, speed_on_output):
                        rst.append(
                            (freq, f"{lbl}_{i+1}",)
                        )

                    for bearing in bearing_dict[i+1]:
                        for freq, lbl in bearing.get_freqs_labels_at(speed):
                            rst.append(
                                (freq, f"{lbl}_{bearing.name}",)
                            )
                speed = speed * stg.ratio
        else:
            n = len(self.stages)
            for i, stg in enumerate(self.stages[::-1]):
                if (choice_bits==-1) or (choice_bits & (1<<(n-i-1))):
                    for freq, lbl in stg.get_freqs_labels_at(speed, speed_on_output):
                        rst.append(
                            (freq, f"{lbl}_{n-i}",)
                        )
                speed = speed / stg.ratio

                # always bearing on input shaft
                # need to make a virtual 1:1 stage for final output
                if (choice_bits==-1) or (choice_bits & (1<<(n-i-1))):
                    for bearing in bearing_dict[n-i]:
                        for freq, lbl in bearing.get_freqs_labels_at(speed):
                            rst.append(
                                (freq, f"{lbl}_{bearing.name}",)
                            )

        # TODO: force regarding input shaft to simplify code

        return rst

class BearingInputStage(int): # cannot use namedtuple(BallBearing, BearingInputStage) because no conversion for namedtuple
    # always the input speed stage
    pass

class BallBearing(DataBase):
    def __init__(self, name: str = None, uuid: str = None, N_balls: int=8, D_ball: float=2, D_pitch: float=12, beta: float=15) -> None:
        super().__init__(name, uuid)
        self.N_balls = N_balls
        self.D_ball = D_ball
        self.D_pitch = D_pitch # = (D_IR+D_OR)/2
        self.beta = beta

    def get_freqs_labels_at(self, speed: float):
        freq = speed / 60
        return [
            (self.bpfo()*freq, 'bpfo',),
            (self.bpfi()*freq, 'bpfi',),
            (self.bsf()*freq, 'bsf',),
            (self.ftf()*freq, 'ftf',),
        ]

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