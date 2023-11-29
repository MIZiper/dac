from nptdms import TdmsFile, TdmsGroup, TdmsChannel
from nptdms import TdmsWriter, RootObject, GroupObject, ChannelObject

from collections import OrderedDict
from . import TimeData
from datetime import datetime, timedelta, timezone

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
            # length = prop['wf_samples']

            r.append(TimeData(name=c.name, y=c.data, dt=dt, y_unit=y_unit, comment=desc))

    return r

def save_tdms(channels: list[TimeData], fpath: str, start_time: datetime, group_name: str="group_1"):
    current_tz = datetime.now().astimezone().tzinfo
    start_time = start_time.replace(tzinfo=current_tz)

    with TdmsWriter(fpath) as tdms_writer:
        root_object = RootObject()
        group_object = GroupObject(group_name)

        for channel in channels:
            channel_object = ChannelObject(group_name, channel.name, channel.y, properties={
                "unit_string": channel.y_unit,
                "wf_xunit_string": "s",
                "wf_xname": "Time",
                "wf_start_time": start_time,
                "wf_increment": channel.dt,
                "wf_samples": channel.length
            })

            tdms_writer.write_segment([
                root_object,
                group_object,
                channel_object,
            ])