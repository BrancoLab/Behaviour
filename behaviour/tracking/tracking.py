import numpy as np
import pandas as pd

from fcutils.file_io.utils import check_file_exists
from fcutils.maths.filtering import median_filter_1d

from fcutils.maths.geometry import calc_distance_between_points_in_a_vector_2d as get_speed_from_xy
from fcutils.maths.geometry import calc_angle_between_points_of_vector_2d as get_dir_of_mvmt_from_xy
from fcutils.maths.geometry import calc_angle_between_vectors_of_points_2d as get_bone_angle
from fcutils.maths.geometry import calc_ang_velocity
from fcutils.maths.utils import derivative
from fcutils.maths.filtering import median_filter_1d

from behaviour.tracking.utils import clean_dlc_tracking
from behaviour.common_coordinates.fisheye import correct_trackingdata_fisheye
from behaviour.common_coordinates.common_coordinates import register_tracking_data



def prepare_tracking_data(tracking_filepath, likelihood_th=0.999,
						median_filter=False, filter_kwargs={},
						fisheye=False, fisheye_args=[],
						common_coord=False, ccm_mtx=None,
						compute=True, smooth_dir_mvmt=True):
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
		:param ccm_mtx: np.array with matrix for common coordinates registration
		:param compute: if true speeds and angles are computed
		:param smooth_dir_mvmt: if true the direction of mvmt is smoothed with a median filt.
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
		likelihoods[bp] = tracking[bp]['likelihood'].values
		tracking[bp].drop('likelihood', axis=1)

	# Median filtering
	if median_filter:
		print("     applying median filter")
		for bp in bodyparts:
			tracking[bp]['x'] = median_filter_1d(tracking[bp]['x'].values, **filter_kwargs)
			tracking[bp]['y'] = median_filter_1d(tracking[bp]['y'].values, **filter_kwargs)

	# Fisheye correction
	if fisheye:
		raise NotImplementedError
		print("     applying fisheye correction")
		if len(fisheye_args) != 3:
			raise ValueError("fish eye correction requires 3 arguments \
						but {} were pased".format(len(fisheye_args)))

		for bp in bodyparts:
			tracking[bp] = correct_trackingdata_fisheye(tracking[bp], *fisheye_args)

	# Reference frame registration
	if common_coord:
		print("     registering to reference space")
		if ccm_mtx is None:
			raise ValueError("ccm_mtx cannot be None")
		 
		for bp in bodyparts:
			xy = np.vstack([tracking[bp]['x'].values, tracking[bp]['y'].values]).T
			corrected_xy = register_tracking_data(xy, ccm_mtx)
			tracking[bp]['x'] = corrected_xy[:, 0]
			tracking[bp]['y'] = corrected_xy[:, 1]


	# Compute speed, angular velocity etc...
	if compute:
		print("     computing speeds and angles")
		for bp in bodyparts:
			x, y = tracking[bp].x.values, tracking[bp].y.values

			tracking[bp]['speed'] = get_speed_from_xy(x, y)

			if not smooth_dir_mvmt:
				tracking[bp]['direction_of_movement'] = get_dir_of_mvmt_from_xy(x, y)	
			else:
				tracking[bp]['direction_of_movement'] = median_filter_1d(get_dir_of_mvmt_from_xy(x, y), kernel=41)	

			tracking[bp]['angular_velocity'] = calc_ang_velocity(tracking[bp]['direction_of_movement'].values)
	
	# Remove low likelihood frames
	for bp, like in likelihoods.items():
		tracking[bp][like < likelihood_th] = np.nan

	return tracking



def compute_body_segments(tracking, segments, smooth_orientation=True):
	""" 
		Given a dictionary of dataframes with tracking and a list of bones (body segments) it computes stuff on the bones
		and returns the results

		:param tracking: dictionary of dataframes with tracking for each bodypart
		:param segments: dict of two-tuples. Keys are the names of the bones and tuple elements the 
				names of the bodyparts that define each bone.
		:param smooth_orientation: bool, if true the bone angles are smoothed with a median filter

	"""
	print("Processing body segments")

	bones = {}
	for bone, (bp1, bp2) in segments.items():

		# get the XY tracking data
		bp1, bp2 = tracking[bp1], tracking[bp2]

		# get angle and ang vel 
		if not smooth_orientation:
			bone_orientation = get_bone_angle(bp1.x.values, bp1.y.values,
											bp2.x.values, bp2.y.values,)
		else:
			bone_orientation = median_filter_1d(get_bone_angle(bp1.x.values, bp1.y.values,
											bp2.x.values, bp2.y.values,), kernel=41)	

		# Get angular velocity
		bone_angvel = np.array(calc_ang_velocity(bone_orientation))

		bones[bone] = pd.DataFrame(dict(
					orientation = bone_orientation, 
					angular_velocity = bone_angvel,
					))
	return bones