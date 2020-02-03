import sys
sys.path.append('./')

import os

from fcutils.file_io.utils import check_file_exists

from utils import open_tdms, get_tdms_group_channels, get_video_params_from_metadata_tdms, get_analog_inputs_clean_dataframe
from behaviour.utilities.signals import get_frames_times_from_squarewave_signal


"""
    Script to check the number of dropped frames in a mantis recording.
"""
def inspect_metadata_file(metadata_tdms):
    """
        It looks at what's inside a metadata tdms file and 
        it checks what the expected size of the video tdms file should be 
        given the # of frames and the frame size.
    """
    # Load metadata frame, get frame size and expected number of frames
    print('\nLoading metadata')
    metadata = open_tdms(metadata_tdms)
    metadata_channels = get_tdms_group_channels(metadata, 'keys')

    print('Found the following groups in metadata:')
    [print("    {}".format(k)) for k in metadata_channels.keys()]

    # Get video params
    video_params = get_video_params_from_metadata_tdms(metadata, is_opened=True)
    print('\nMetadata video parameters:')
    [print("    {}:{}".format(k,v)) for k,v in video_params.items()]
    print("\n\n")

    # Check height match
    actual_height = metadata.as_dataframe()["/'keys'/'IMAQdxActualHeight'"][0]
    if actual_height != video_params['height']:
        raise NotImplementedError("Real and actual frame height mismatch. Haven't dealt with this case yet, sorry.")

    # Get the expected file size
    expected_nbytes = (video_params['width'] * video_params['height']) * video_params['last']
    return expected_nbytes


def check_mantis_dropped_frames(experiment_folder, camera_name, experiment_name, 
                        skip_analog_inputs=False,
                        camera_triggers_channel=None):
    """
        Checks if Mantis dropped any frames by:
            1) checking if the size of the video .tdms is what you'd expect given the number of frames and frame size
                stored in the metadata .tdms file
            2) [optional] by looking at the camera triggers recorded as analog input (AI) and checking
                if it matches what's reported on the metadata file.

        :param experiment_folder: str, path to where the data are stored
        :param camera_name: name of the camera used in mantis, used to find video file
        :param experiment_name: name of the experiment used in matnis, used to find metadata file
        :param skip_analog_inputs: bool if true step (2) above is skipped
        :param camera_triggers_channel: str, name of the analog inputs file's channel which recorded the camera triggers
    """

    # ---------- Check video tdms file has the expected number of bytes ---------- #
    # Get file paths and check they exist
    video_tdms = os.path.join(experiment_folder, camera_name+'.tdms')
    metadata_tdms = os.path.join(experiment_folder, camera_name+'meta.tdms')

    if not skip_analog_inputs:
        analog_inputs_tdms = os.path.join(experiment_folder, experiment_name+'(0).tdms')
        files =  [video_tdms, metadata_tdms, analog_inputs_tdms]
    else:
        files = [video_tdms, metadata_tdms]

    for f in files:
        check_file_exists(f, raise_error=True)

    # Get expected n bytes
    expected_nbytes = inspect_metadata_file(metadata_tdms)

    # Check if size of video file is correct
    videofile_size = os.path.getsize(video_tdms)
    if videofile_size != expected_nbytes:
        frame_size = video_params['width'] * video_params['height']
        if abs(videofile_size - expected_nbytes) < frame_size:
            if videofile_size - expected_nbytes > 0:
                s = 'bigger'
            else: 
                s = 'smaller'
            print('Video file {} than expected size. Difference smaller than the size of one frame [{} bytes vs {}].\nSo no frames lost?'.format(s, videofile_size - expected_nbytes, frame_size))
        else:
            raise ValueError("Expected video file to have {} bytes, found {} instead".format(videofile_size, expected_nbytes))
    else:
        print("File size as expected given {} frames. 0 frames lost!".format(video_params['last']))

    # --------------------- INSPECT N FRAMES IN ANALOG INPUT --------------------- #
    if not skip_analog_inputs
        # Load analog inputs
        inputs = get_analog_inputs_clean_dataframe(analog_inputs_tdms, is_opened=False)
        n_frames = len(get_frames_times_from_squarewave_signal(inputs[camera_triggers_channel].values, debug=True))

        if n_frames != video_params['last']:
            raise ValueError("Number of frames in the frames AI ({}) is different than the expected number if frames ({})".format(n_frames, video_params['last']))
        else:
            print("number of frames in the analog input is correct, no frames dropped.")
    else:
        print("Skipping analysis of recorded camera triggers in analog input file.")


if __name__ == "__main__":
    experiment_folder = 'Z:\\swc\\branco\\rig_photometry\\Mantis_test\\test'
    camera_name = 'FP_behav_camera'
    experiment_name='FP_just_behav'
    camera_triggers_channel='FP_behav_camera_triggers_reading'

    check_mantis_dropped_frames(experiment_folder, camera_name, experiment_name, camera_triggers_channel)