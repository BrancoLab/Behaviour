import numpy as np
import pandas as pd

from fcutils.file_io.utils import check_file_exists
from fcutils.maths.filtering import median_filter_1d

from fcutils.maths.geometry import calc_distance_between_points_in_a_vector_2d as get_speed_from_xy
from fcutils.maths.geometry import calc_angle_between_points_of_vector_2d as get_dir_of_mvmt_from_xy
from fcutils.maths.geometry import calc_angle_between_vectors_of_points_2d as get_bone_angle
from fcutils.maths.geometry import calc_ang_velocity
from fcutils.maths.geometry import calc_distance_between_points_two_vectors_2d as get_bone_length
from fcutils.maths.utils import derivative
from fcutils.maths.filtering import median_filter_1d

from behaviour.tracking.utils import clean_dlc_tracking
from behaviour.common_coordinates.fisheye import correct_trackingdata_fisheye
from behaviour.common_coordinates.common_coordinates import register_tracking_data



def prepare_tracking_data(tracking_filepath, 
						likelihood_th=0.999,
						median_filter=False, filter_kwargs={},
						fisheye=False, fisheye_args=[],
						common_coord=False, ccm_mtx=None,
						compute=True, 
						smooth_dir_mvmt=True,
						interpolate_nans=False,
						verbose=False):
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
		:param interpolate_nans: if true it removes nans from the tracking data by linear interpolation
	"""

	# Load the tracking data
	check_file_exists(tracking_filepath, raise_error=True)
	if '.h5' not in tracking_filepath:
		raise ValueError("Expected .h5 in the tracking data file path")
	
	if verbose:
		print('Processing: {}'.format(tracking_filepath))
	tracking, bodyparts = clean_dlc_tracking(pd.read_hdf(tracking_filepath))

	# Get likelihood and XY coords
	likelihoods = {}
	for bp in bodyparts:
		likelihoods[bp] = tracking[bp]['likelihood'].values
		tracking[bp].drop('likelihood', axis=1)

	# Median filtering
	if median_filter:
		if verbose:
			print("     applying median filter")
		for bp in bodyparts:
			tracking[bp]['x'] = median_filter_1d(tracking[bp]['x'].values, **filter_kwargs)
			tracking[bp]['y'] = median_filter_1d(tracking[bp]['y'].values, **filter_kwargs)

	# Fisheye correction
	if fisheye:
		raise NotImplementedError
		if verbose:
			print("     applying fisheye correction")
		if len(fisheye_args) != 3:
			raise ValueError("fish eye correction requires 3 arguments \
						but {} were pased".format(len(fisheye_args)))

		for bp in bodyparts:
			tracking[bp] = correct_trackingdata_fisheye(tracking[bp], *fisheye_args)

	# Reference frame registration
	if common_coord:
		if verbose:
			print("     registering to reference space")
		if ccm_mtx is None:
			raise ValueError("ccm_mtx cannot be None")
		 
		for bp in bodyparts:
			xy = np.vstack([tracking[bp]['x'].values, tracking[bp]['y'].values]).T
			corrected_xy = register_tracking_data(xy, ccm_mtx)
			tracking[bp]['x'] = corrected_xy[:, 0]
			tracking[bp]['y'] = corrected_xy[:, 1]

	# Remove nans
	if interpolate_nans:
		for bp in bodyparts:
			# Check how many nans
			track = tracking[bp].copy()
			track[like[bp] < likelihood_th] = np.nan
			number_of_nans = tracking[bp]['x'].isna().sum()
			if number_of_nans >= len(track/100):
				print(f'Found > 1% of frames with nan value (i.e. bad tracking) for body part {bp}'+
						f'[{number_of_nans} frames out of {len(track)}]\n'+
						'Perhaps consider improving tracking quality ?')
			tracking[bp] = track.interpolate(axis=1)

	# Compute speed, angular velocity etc...
	if compute:
		if verbose:
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

		# Get bone length [first remove nans to allow computation]
		bp1_tr, bp2_tr = np.array([bp1.x.values, bp1.y.values]).T, np.array([bp2.x.values, bp2.y.values]).T
		nan_idxs = list(np.where(np.isnan(bp1_tr[:, 0]))[0])  + \
					list(np.where(np.isnan(bp1_tr[:, 1]))[0]) + \
					list(np.where(np.isnan(bp2_tr[:, 0]))[0]) + \
					list(np.where(np.isnan(bp2_tr[:, 1]))[0])

		bone_length = get_bone_length(np.nan_to_num(bp1_tr), np.nan_to_num(bp1_tr))
		bone_length[nan_idxs] = np.nan # replace nans

		# Put everything together
		bones[bone] = pd.DataFrame(dict(
					orientation = bone_orientation, 
					angular_velocity = bone_angvel,
					bone_length = bone_length,
					))
	return bones