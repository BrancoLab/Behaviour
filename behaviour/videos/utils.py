from tqdm import tqdm
import cv2
import numpy as np

from fcutils.file_io.utils import check_file_exists
from fcutils.video.utils import get_video_params, get_cap_from_file


def get_background_from_video(videopath, start_frame=0, avg_over=10):
    """
        Extracts background by averaging across video frames

        :param videopath: str, path to video to analyse
        :param start_frame: int, frame to start at 
        :param avg_over: int, a frame every N is used for analysis to speed things up
    """

    check_file_exists(videopath, raise_error=True)

    # Open video and get params
    cap = get_cap_from_file(videopath)
    nframes, width, height, fps = get_video_params(cap)

    # Start at selected frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Get background by averaging every N frames
    background = np.zeros((height, width))
    every_other = int(num_frames / avg_over)
    j = 0

    for i in tqdm(range(num_frames)):

        if i % every_other == 0:
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vid.read()  # get the frame

            if ret:
                # store the current frame in as a numpy array
                background += frame[:, :, 0]
                j += 1
            else:
                break

    background = (background / (j)).astype(np.uint8)
    cap.release()

    return background
