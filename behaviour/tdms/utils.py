import pandas as pd
from collections import OrderedDict

from nptdms import TdmsFile

from fcutils.file_io.utils import check_file_exists, check_create_folder


# ---------------------------------------------------------------------------- #
#                  UTILITY FUNCTIONS SPECIFIC FOR MANTIS FILES                 #
# ---------------------------------------------------------------------------- #


def get_video_params_from_metadata_tdms(metadata, is_opened=False):
    """
        Gets video params [fps, frame width and height and number of frames]
        from a Mantis video metadata tdms file. 

        :param metadata: instance of TdmsFile of path to a tdms metadata file
        :param is_opened: bool, if the path is passed, is_opened should be false so that the tdms file should be opened
    """
    if not is_opened:
        metadata = open_tdms(metadata)

    return {n: v for n, v in metadata.object().properties.items()}


def get_analog_inputs_clean_dataframe(analog_inputs, is_opened=False, overwrite=False, save_df=True):
    """
        Returns a dataframe with data from mantis analog inputs file, after cleaning 
        up the names of the channels. 

        :param analog_inputs: instance of pd.DataFrame of path to a tdms analog_inputs file
        :param overwrite: bool, if false it avoid overwriting a previously saved dataframe
        :param is_opened: bool, if the path is passed, is_opened should be false so that the tdms file should be opened
        :param save_df: if True it will save the df to file to avoid loading everytime
    """
    if not is_opened:
        df_path = analog_inputs.split(".")[0]+".h5"
        if check_file_exists(df_path) and not overwrite:
            return pd.read_hdf(df_path, key='hdf')        
        analog_inputs = open_tdms(analog_inputs, as_dataframe=True)[0]

    else:
        if not isinstance(analog_inputs, pd.DataFrame):
            raise ValueError(
                "Opened analog inputs file should be passed as a dataframe"
            )

    clean_columns = [
        c.strip("'0'").strip("'").strip("/").strip("'") for c in analog_inputs.columns
    ]
    analog_inputs.columns = clean_columns
    
    if not is_opened and save_df:
        analog_inputs.astype('float64').to_hdf(df_path, key="hdf")

    return analog_inputs


def get_analog_inputs_clean(analog_inputs, is_opened=False):
    """
        Returns a dictionary with data from mantis analog inputs file, after cleaning 
        up the names of the channels. 

        :param analog_inputs: instance of TdmsFile of path to a tdms analog_inputs file
        :param is_opened: bool, if the path is passed, is_opened should be false so that the tdms file should be opened
    """
    if not is_opened:
        analog_inputs = open_tdms(analog_inputs)

# ---------------------------------------------------------------------------- #
#                             GENERAL I/O FUNCTIONS                            #
# ---------------------------------------------------------------------------- #
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
    if ".tdms" not in tdms_path:
        raise ValueError("The file passed doesn't seem to be a TDMS: " + tdms_path)

    if memmap_dir is None or not memmap_dir:
        tdms = TdmsFile(tdms_path)
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


def get_tdms_group_channels(tdms, group):
    """
        Returns the channels that belong to a group in tdms file

        :param tdms: instance of TdmsFile
        :param group: string, name of the group whose channels you neeed
    """
    groups = get_tdms_groups(tdms)
    if group not in groups:
        raise ValueError("Group not found!")

    channels = tdms.group_channels(group)
    return {ch.path.split("'/'")[-1][:-1]: ch for ch in channels}


def get_tdms_channel_data(tdms, channel, num):
    return tdms.channel_data(channel, str(num))
