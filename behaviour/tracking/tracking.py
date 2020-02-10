import numpy as np
import pandas as pd

from fcutils.file_io.utils import check_file_exists
from fcutils.maths.filtering import median_filter_1d

from utils import clean_dlc_tracking, get_speed_from_xy, get_dir_of_mvmt_from_xy, get_ang_vel_from_xy
from behaviour.common_coordinates.fisheye import correct_trackingdata_fisheye
from behaviour.common_coordinates.common_coordinates import register_tracking_data



def prepare_tracking_data(tracking_filepath, likelihood_th=0.999,
                        median_filter=False, filter_kwargs={},
                        fisheye=False, fisheye_args=[],
                        common_coord=False, common_coord_args=[],
                        compute=True):
    """
        Loads, cleans and filters tracking data from dlc.
        Also handles fisheye correction and registration to common coordinates frame.
        Can be used to compute speeds and angles for each bp.

        :param tracking_filepath: path to file to process
        :param likelihood_th: float, frames with likelihood < thresh are nanned
        :param median_filter: if true the data are filtered before the processing
        :param filter_kwargs: arguments for median filtering func
        :param fisheye: if true fish eye correction is applied
        :param fisheye_args: arguments for fisheye correction func
        :param common_coord: if true common coordinates referencing is done
        :param common_coord_args: arguments for common coordinates registration
        :param compute: if true speeds and angles are computed
    """

    # Load the tracking data
    check_file_exists(tracking_filepath, raise_error=True)
    if '.h5' not in tracking_filepath:
        raise ValueError("Expected .h5 in the tracking data file path")
    
    print('Processing: {}'.format(tracking_filepath))
    tracking, bodyparts = clean_dlc_tracking(pd.read_hdf(tracking_filepath))

    # Get likelihood and XY coords
    likelihoods = {}
    for bp in bodyparts:
        likelihoods[bp] = tracking[bp].values[:, 2]
        tracking[bp].drop('likelihood')

    # Median filtering
    if median_filter:
        print("     applying median filter")
        for bp in bodyparts:
            tracking[bp]['x'] = median_filter_1d(tracking[bp]['x'], **filter_kwargs)
            tracking[bp]['y'] = median_filter_1d(tracking[bp]['y'], **filter_kwargs)

    # Fisheye correction
    if fisheye:
        print("     applying fisheye correction")
        if len(fisheye_args) != 3:
            raise ValueError("fish eye correction requires 3 arguments \
                        but {} were pased".format(len(fisheye_args)))

        for bp in bodyparts:
            tracking[bp] = correct_trackingdata_fisheye(tracking[bp], *fisheye_args)

    # Reference frame registration
    if common_coord:
        print("     registering to reference space")
        if len(common_coord_args) != 3:
            raise ValueError("reference frame registation requires 3 arguments \
                but {} were passed".format(len(common_coord_args)))
         
        for bp in bodyparts:
            tracking[bp] = register_tracking_data(tracking[bp], *common_coord_args)
    
    # Remove low likelihood frames
    for bp, like in likelihoods.items():
        tracking[bp][like < likelihood_th] = np.nan

    # Compute speed, angular velocity etc...
    if compute:
        print("     computing speeds and angles")
        for bp in bodyparts:
            tracking[bp]['speed'] = get_speed_from_xy(tracking[bp].values[:, :2])

            tracking[bp]['direction_of_movement'] = get_dir_of_mvmt_from_xy(np.vstack([tracking[bp]['x'], tracking[bp]['y']]).T)

            tracking[bp]['angular_velocity'] = get_ang_vel_from_xy(np.vstack([tracking[bp]['x'], tracking[bp]['y']]).T)
    
    return tracking



def compute_body_segments(tracking, segments):
	""" 
		Given a dataframe with tracking and a list of bones (body segments) it computes stuff on the bones
		and returns the results as a dataframe

		:param tracking: hierarchical dataframe
		:param segments: dict with keys as the names of the body segments and values as
			tuples of bodyparts connected by the semgents

	"""
	raise NotImplementedError("Find a way to return it as a hierarchical DF like the tracking one")

	for bone, (bp1, bp2) in segments.items():
		segment = {}
		segkey['name'], segkey['bp1'], segkey['bp2'] = bone, bp1, bp2


		# get the XY tracking data
		bp1, bp2 = tracking[bp1]['x', 'y'], tracking[bp2]['x', 'y']

		# get angle and ang vel 
		bone_orientation = np.array(calc_angle_between_vectors_of_points_2d(bp1.T, bp2.T))

		# Get angular velocity
		bone_angvel = np.array(get_ang_vel_from_xy(angles=bone_orientation))

        # TODO Find a way to put the results together somehow