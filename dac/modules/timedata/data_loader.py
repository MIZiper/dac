"""Provides functions for loading and saving TimeData from/to TDMS files.
"""

from nptdms import TdmsFile, TdmsGroup, TdmsChannel
from nptdms import TdmsWriter, RootObject, GroupObject, ChannelObject

from collections import OrderedDict
from . import TimeData
from datetime import datetime, timedelta, timezone

def load_tdms(fpath) -> list[TimeData]:
    """Loads data from a TDMS file and converts channels to TimeData objects.

    Iterates through all groups and channels in the TDMS file, extracting
    data and relevant properties to create TimeData objects.

    Parameters
    ----------
    fpath : str
        The file path to the TDMS file.

    Returns
    -------
    list[TimeData]
        A list of TimeData objects, each representing a channel from
        the TDMS file.
    """

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
    """Saves a list of TimeData objects to a TDMS file.

    Each TimeData object is written as a channel within the specified group
    in the TDMS file. Includes metadata such as units, start time,
    and sampling interval (dt).

    Parameters
    ----------
    channels : list[TimeData]
        A list of TimeData objects to save.
    fpath : str
        The file path where the TDMS file will be saved.
    start_time : datetime
        A datetime object representing the start time of the
        measurement. The current system timezone is applied to it.
    group_name : str, default "group_1"
        The name of the group under which channels will be saved
        in the TDMS file.
    """
    
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