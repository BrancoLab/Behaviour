import pandas as pd
import numpy as np
import os

from fcutils.maths.geometry import calc_distance_between_points_in_a_vector_2d
from fcutils.maths.geometry import calc_angle_between_points_of_vector, calc_ang_velocity
from fcutils.maths.geometry import calc_angle_between_vectors_of_points_2d

def get_scorer_bodyparts(tracking):
    """
        Given the tracking data hierarchical df from DLC, return
        the scorer and bodyparts names
    """
    first_frame = posedata.iloc[0]
    try:
        bodyparts = first_frame.index.levels[1]
        scorer = first_frame.index.levels[0]
    except:
        raise NotImplementedError("Make this return something helpful when not DLC df")

    return scorer, bodyparts


def clean_dlc_tracking(tracking):
    """
        Given the tracking data hierarchical df from DLC, 
        returns a simplified version of it. 
    """
    scorer, bodyparts = get_scorer_bodyparts(tracking)
    return tracking[scorer], bodyparts
    

def get_speed_from_xy(xy):
    return calc_distance_between_points_in_a_vector_2d(xy)

def get_dir_of_mvmt_from_xy(xy):
    return calc_angle_between_points_of_vector(xy)

def get_orientation_from_two_xy(xy1, xy2):
    np.array(calc_angle_between_vectors_of_points_2d(xy1.T, xy1.T))

def get_ang_vel_from_xy(xy=None, angles=None):
    if xy is not None:
        return calc_ang_velocity(get_dir_of_mvmt_from_xy(xy))
    elif angles is not None:
        return calc_ang_velocity(angles)
    else:
        raise ValueError("No data passed!")