import cv2
import numpy as np
import glob
import os
import pandas as pd

def invert_fisheye_map(fisheye_map_location, inverse_fisheye_map_location):
    '''Go from a normal opencv fisheye map to an inverted one, so coordinates can be transformed'''

    print('creating inverse fisheye map')

    # load regular fisheye mapping
    original_fisheye_map = np.load(fisheye_map_location)

    # invert maps
    original_fisheye_map[original_fisheye_map < 0] = 0

    maps_x_orig = original_fisheye_map[:, :, 0]
    maps_x_orig[maps_x_orig > 1279] = 1279
    maps_y_orig = original_fisheye_map[:, :, 1]
    maps_y_orig[maps_y_orig > 1023] = 1023

    map_x = np.ones(original_fisheye_map.shape[0:2]) * np.nan
    map_y = np.ones(original_fisheye_map.shape[0:2]) * np.nan
    for x in range(original_fisheye_map.shape[1]):
        for y in range(original_fisheye_map.shape[0]):
            map_x[maps_y_orig[y, x], maps_x_orig[y, x]] = x
            map_y[maps_y_orig[y, x], maps_x_orig[y, x]] = y

    grid_x, grid_y = np.mgrid[0:original_fisheye_map.shape[0], 0:original_fisheye_map.shape[1]]
    valid_values_x = np.ma.masked_invalid(map_x)
    valid_values_y = np.ma.masked_invalid(map_y)

    valid_idx_x_map_x = grid_x[~valid_values_x.mask]
    valid_idx_y_map_x = grid_y[~valid_values_x.mask]

    valid_idx_x_map_y = grid_x[~valid_values_y.mask]
    valid_idx_y_map_y = grid_y[~valid_values_y.mask]

    map_x_interp = interpolate.griddata((valid_idx_x_map_x, valid_idx_y_map_x), map_x[~valid_values_x.mask],
                                        (grid_x, grid_y), method='linear').astype(np.uint16)
    map_y_interp = interpolate.griddata((valid_idx_x_map_y, valid_idx_y_map_y), map_y[~valid_values_y.mask],
                                        (grid_x, grid_y), method='linear').astype(np.uint16)

    inverse_fisheye_map = np.zeros((map_x_interp.shape[0], map_x_interp.shape[1], 2)).astype(np.uint16)
    inverse_fisheye_map[:, :, 0] = map_x_interp
    inverse_fisheye_map[:, :, 1] = map_y_interp

    np.save(inverse_fisheye_map_location, inverse_fisheye_map)

    return registration



def extract_coordinates_with_dlc(video_file, registration, body_parts, inverse_fisheye_map_location, x_offset=0, y_offset=0):
    '''
    Extract coordinates for each frame, given a video and DeepLabCut network
    Then apply inverse fisheye correction and the affine transformation to put the data in common coordinate space

    video_file: file location of the video being tracked
    registration e.g. as from Register_and_Display_Behavior.py
    body_parts: DLC tracted body parts e.g. ['foot','back','snout']
    inverse_fisheye_map_location: as generated in invert_fisheye_map
    '''

    # read the coordinates file from DLC
    coordinates_file = glob.glob(os.path.dirname(video_file) + '\\*.h5')[0]
    DLC_network = os.path.basename(coordinates_file)
    DLC_network = DLC_network[DLC_network.find('Deep'):-3]
    DLC_dataframe = pd.read_hdf(coordinates_file)
    coordinates = {}

    # fisheye correct the coordinates
    # registration = invert_fisheye_map(registration[3], inverse_fisheye_map_location)
    inverse_fisheye_maps = np.load(inverse_fisheye_map_location)

    # loop over each marked body part
    for i, body_part in enumerate(body_parts):

        # initialize coordinates
        coordinates[body_part] = np.zeros((2, len(DLC_dataframe[DLC_network][body_part]['x'].values)))

        # extract coordinates
        for j, axis in enumerate(['x', 'y']):
            coordinates[body_part][j] = DLC_dataframe[DLC_network][body_part][axis].values
            coordinates[body_part][j] = DLC_dataframe[DLC_network][body_part][axis].values

        # convert original coordinates to registered coordinates
        coordinates[body_part][0] = inverse_fisheye_maps[
                                        coordinates[body_part][1].astype(np.uint16) + y_offset, coordinates[body_part][
                                            0].astype(np.uint16) + x_offset, 0] - x_offset
        coordinates[body_part][1] = inverse_fisheye_maps[
                                        coordinates[body_part][1].astype(np.uint16) + y_offset, coordinates[body_part][
                                            0].astype(np.uint16) + x_offset, 1] - y_offset

        # affine transform to match model arena
        transformed_points = np.matmul(np.append(registration[0], np.zeros((1, 3)), 0),
                                       np.concatenate((coordinates[body_part][0:1], coordinates[body_part][1:2],
                                                       np.ones((1, len(coordinates[body_part][0])))), 0))

        coordinates[body_part][0] = transformed_points[0, :]
        coordinates[body_part][1] = transformed_points[1, :]

    return coordinates
