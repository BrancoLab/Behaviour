import pandas as pd
from collections import OrderedDict

from nptdms import TdmsFile

from fcutils.file_io.utils import check_file_exists, check_create_folder

def tdms_as_dataframe(tdms, time_index=False, absolute_time=False):
    """
    Converts the TDMS file to a DataFrame

    :param tdms: an instance of TdmsFile
    :param time_index: Whether to include a time index for the dataframe.
    :param absolute_time: If time_index is true, whether the time index
        values are absolute times or relative to the start time.
    """
    keys = []  # ? also return all the columns as well
    dataframe_dict = OrderedDict()
    for key, value in tdms.objects.items():
        keys.append(key)
        if value.has_data:
            index = value.time_track(absolute_time) if time_index else None
            dataframe_dict[key] = pd.Series(data=value.data, index=index)
    return pd.DataFrame.from_dict(dataframe_dict), keys


def open_tdms(tdms_path, memmap_dir=False, as_dataframe=False):
    """
        Open tdms as either TdmsFile object or dataframe

        :param tdms_path: path to file to open
        :param memmap_dir: path to a directory to memmap large files
        :param as_dataframe: bool, if true a dataframe is returned
    """
    check_file_exists(tdms_path, raise_error=True)
    if '.tdms' not in tdms_path:
        raise ValueError("The file passed doesn't seem to be a TDMS: "+tdms_path)

    if memmap_dir is None:
        tdms =  TdmsFile(tdms_path)
    else:
        check_create_folder(memmap_dir, raise_error=True)
        tdms = TdmsFile(tdms_path, memmap_dir=memmap_dir)

    if as_dataframe:
        return tdms_as_dataframe(tdms)
    else:
        return tdms


def get_tdms_groups(tdms):
    """
        Returns the groups in a tdms file

        :param tdms: instance of TdmsFile
    """
    return tdms.groups()

def get_tdms_group_channels(tdms, group)
    """
        Returns the channels that belong to a group in tdms file

        :param tdms: instance of TdmsFile
        :param group: string
    """
    groups = get_tdms_groups(tdms)
    if group not in groups:
        raise ValueError("Group not found!")

    return tdms_df.group_channels(group)

def get_tdms_channel_data(tdms, channel, num):
    return tdms_df.channel_data(channel, str(num))

