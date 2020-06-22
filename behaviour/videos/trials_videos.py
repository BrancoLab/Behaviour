import numpy as np
import multiprocessing as mp
import os
import cv2

from fcutils.file_io.utils import get_file_name
from fcutils.video.utils import get_cap_from_file, get_cap_selected_frame, get_video_params, open_cvwriter


def make_trials_videos(session_video, stimuli, save_folder=None, 
                            n_sec_pre=5, n_sec_pos=15, stim_duration_sec=9):
    """
        Creates a video with the trials for one session. 
        Id adds some text to say which trial it is and a circle to signal when the stimulus is on.

        :param  session_video: str, path to video to take the frames from
        :param stimuli: list or 1d numpy array with stimuli onset times (in number of frames from start of vid)
        :param save_folder: str, optional. Path to folder where the video will be saved
        :param n_sec_pre: number of seconds before each stimulus to keep in each trial's clip
        :param n_sec_post: number of seconds after each stimulus to keep in each trial's clip
        :param stim_duration_sec: duration of the stimulus in seconds.

    """
    if save_folder is None:
        save_folder = os.path.split(session_video)[0]
    videoname = get_file_name(session_video)
    save_path = os.path.join(save_folder, videoname+'_trials.mp4')

    # Open video
    videocap = get_cap_from_file(session_video)
    nframes, width, height, fps, _ = get_video_params(videocap)
    writer = open_cvwriter(save_path, w=width, h=height, framerate=fps, iscolor=True)

    # Prep some vars
    n_frames_pre = n_sec_pre * fps
    n_frames_pos = n_sec_pos * fps

    # Vars for text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50, 50)
    fontScale              = 1
    lineType               = 2
    text_color             = (30, 220, 30)

    # Vars for circle that signals when the stim is on
    circle_pos = (700, 75)
    circle_radius = 50
    circle_color = (0, 255, 0)

    # Loop over stimuli
    for stimn, stim in enumerate(stimuli):
        for framen in np.arange(stim-n_frames_pre, stim+n_frames_pos):
            frame = get_cap_selected_frame(videocap, int(framen))

            if frame is None: break

            if framen >= stim and framen <= stim+stim_duration_sec*fps:
                cv2.circle(frame, circle_pos, circle_radius, circle_color, -1)

            cv2.putText(frame, f'Stim {stimn} of {len(stimuli)-1}', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                text_color,
                lineType)

            writer.write(frame.astype(np.uint8))
    writer.release()

def make_videos_in_parallel(videos, stimuli, **kwargs):
    """
        Facilitates the creation of stimuli clips in parallel (multiprocessing).

        :param videos: list of string with paths to each video to process
        :param stimuli: list of lists with stimuli onsets for each session
        :param kwargs: other arguments to pass to make_trials_videos
    """
    raise NotImplementedError('This function has a bug: it needs a better way to pass args to make_trials_videos')
    pool = mp.Pool(mp.cpu_count()-2)
    pool.map(make_trials_videos, [(vid, ai, kwargs) for vid, ai in zip(videos, stimuli)])
    pool.close()