from nptdms import TdmsFile, TdmsGroup, TdmsChannel
from collections import OrderedDict
from . import TimeData

def load_tdms(fpath) -> list[TimeData]:
    r = []
    f = TdmsFile(fpath, read_metadata_only=False, keep_open=False)
    for g in f.groups():
        g: TdmsGroup
        for c in g.channels():
            c: TdmsChannel
            prop: OrderedDict = c.properties

            gain = float(prop['Gain'])
            offset = float(prop['Offset'])
            y_unit = prop['Unit']
            desc = prop['Description']
            x_unit = prop['wf_xunit_string']
            dt = float(prop['wf_increment'])
            length = prop['wf_samples']

            r.append(TimeData(name=c.name, y=c.data, dt=dt, y_unit=y_unit, comment=desc))

    return r