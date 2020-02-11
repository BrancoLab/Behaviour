import numpy as np
import cv2
import os

from fcutils.file_io.utils import check_file_exists, check_create_folder, listdir

# ---------------------------------------------------------------------------- #
#                             FISH EYE CALIBRATION                             #
# ---------------------------------------------------------------------------- #
"""
    The original code was written by Philip Shamash and wa adapted from
    https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
    see Kenneth Jiang's blog post above for helpful explanation
""" 

class FishEyeCalibration:
    def __init__(self, calibration_images_folder,  camera_name, 
                checkerboard_shape, dark_threshold, images_extention='.png',
                save_folder=None, save_inverted_map=True):
        """ 
            Initialise and run fish eye calibration
        
            :param calibration_images_folder: path to folder with calibration images
            :param images_extention: file type of images
            :param camera_name: str, used to save the maps
            :param checkerboard_shape: tuple with  #size of the checkerboard (# of vertices in each dimension, not including those on the edge)
            :param dark_threshold: int, The algorithm is finnicky and likes saturation, so set pixels with values below the dark_threshold to black
            :param save_folder: str, path to a directory where to save the output
            :save_inverted_map: bool, if true an inverted map used for tracking data correction is saved
        """
        # ----------------------------------- setup ---------------------------------- #
        check_create_folder(calibration_images_folder, raise_error=True)
        self.calibration_images = [f for f in listdir(calibration_images_folder) if images_extention in f]
        print('Found ' + str(len(self.calibration_images)) + ' images for fisheye calibration.')

        if save_folder is None:
            self.save_folder = calibration_images_folder
        else:
            check_create_folder(save_folder)
            self.save_folder = save_folder

        self.camera_name = camera_name
        self.checkerboard_shape = checkerboard_shape
        self.dark_threshold = dark_threshold
        self.save_inverted_map = save_inverted_map

    def get_map(self, inspect_results=True):   
        """
            Main function used to compute correction maps from correction images
        """
        # Find checkerboard in calibration images
        objpoints, imgpoints, calib_image, _img_shape = self.find_checkerboard_corners()

        # Get transform matrices
        K, D, DIM= self.get_calibration_matrices(objpoints, imgpoints, calib_image, _img_shape)

        # Check quality of correction
        maps = self.test_calibration(K, D, DIM, calib_image)

        # Save maps
        maps_save_path = self.save_fisheye_maps(maps)

        if self.save_inverted_map:
            inverted_maps = self.invert_fisheye_map(map_save_path)
            inverted_maps_save_path = self.save_fisheye_maps(inverted_maps, inverted=True)
        else:
            inverted_maps = inverted_maps_save_path = None
        
        return (maps, maps_save_path), (inverted_maps, inverted_maps_save_path)

    def find_checkerboard_corners(self):
        """
            Loops over the calibration images and tries to find the checkerboard pattern in each of them
        """
        # Initialise variables
        CHECKERFLIP = tuple(np.flip(self.checkerboard_shape,0))
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        objp = np.zeros((1, self.checkerboard_shape[0]*self.checkerboard_shape[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:self.checkerboard_shape[0], 0:self.checkerboard_shape[1]].T.reshape(-1, 2)

        _img_shape = None
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # Loop over images
        for fn, fname in enumerate(self.calibration_images):
            print("Processing image {} of {}".format(fn+1, len(self.calibration_images)))
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."
            
            #increase contrast
            calib_image_pre = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY)
            calib_image = calib_image_pre
            calib_image[calib_image<dark_threshold] = 0

            # Show result
            cv2.imshow('calibration image',calib_image)
            cv2.waitKey(3)

            # Find the chess board corners (takes a while)
            ret, corners = cv2.findChessboardCorners(calib_image, self.checkerboard_shape, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK)

            # If found, add object points, image points (after refining them)
            if ret == True:
                print("     "+ os.path.split(fname)[-1] + ': successfully identified corners')

                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(calib_image,corners,(11,11),(-1,-1),subpix_criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(calib_image, CHECKERBOARD, corners2, ret)
                cv2.imshow('calibration image', calib_image)
                cv2.waitKey(500)
            else:
                print("     "+ os.path.split(fname)[-1] + ': failed to identify corners')

        return objpoints, imgpoints, calib_image, _img_shape
    

    def get_calibration_matrices(sel, objpoints, imgpoints, calib_image, _img_shape):
        """
            Computs the map given the results of 'find_checkerboard_corners'
        """
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW #+cv2.fisheye.CALIB_CHECK_COND
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                calib_image.shape[::-1],
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        print("found " + str(N_OK) + " valid images for calibration")
        print("DIM=" + str(_img_shape[::-1]))
        print("K=np.array(" + str(K.tolist()) + ")")
        print("D=np.array(" + str(D.tolist()) + ")")

        return np.array(K), np.array(D), _img_shape[::-1]

    def test_calibration(self, K, D, DIM, calib_image):
        """ Applies fisheye correction to the calibration images to test quality of correction. """
        # Test calibration
        for img_path in self.calibration_images:
            img = cv2.imread(img_path)

            # correct image
            h,w = img.shape[:2]
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
            assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
            dim2 = dim1 #dim2 is the dimension of remapped image
            dim3 = dim2 #dim3 is the dimension of final output image
            
            # K, dim2 and balance are used to determine the final K used to un-distort image -- balance = 1 retains all pixels
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim2, np.eye(3), balance=1)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            cv2.imshow("correction -- before and after", img)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break 
            cv2.imshow("correction -- before and after", undistorted_img)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break 
        
        # make map
        maps = np.zeros((calib_image.shape[0],calib_image.shape[1],3)).astype(np.int16)
        maps[:,:,0:2] = map1
        maps[:,:,2] = map2
        return maps

    def save_fisheye_maps(self, maps, inverted=False):
        if not inverted:
            save_path = os.path.join(self.save_folder, '{}_fisheye_maps.npy'.format(self.camera_name))
        else:
            save_path = os.path.join(self.save_folder, '{}_inverted_fisheye_maps.npy'.format(self.camera_name))

        np.save(save_path, maps)
        return save_path

    def invert_fisheye_map(self, fisheye_map_filepath):
        '''
            Go from a normal opencv fisheye map to an inverted one that can be used 
            to correct fisheye distortions from tracking data
        '''

        # load regular fisheye mapping
        check_file_exists(fisheye_map_filepath, raise_error=True)
        original_fisheye_map = np.load(fisheye_map_filepath)

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

        return inverse_fisheye_map





# ---------------------------------------------------------------------------- #
#                   OTHER USEFUL FUNCTIONS RELATED TO FISHEYE                  #
# ---------------------------------------------------------------------------- #
def correct_trackingdata_fisheye(tracking, inverse_fisheye_map_file, x_offset, y_offset):
    """
        Given a pd.DataFrame with 'x' and 'y' coordinates, and an inverted fish eye map,
        applies fisheye correction to coordinates.

        :param inverse_fisheye_map_file: path to inverted fish eye map file
        :param tracking: pandas dataframe with tracking data for a bodypart, needs to have x and y columns
        :param x_offset: int if fish eye map was computed for the whole frame but 
                                    the tracking is from a video with samaller frame
        :param y_offset: int if fish eye map was computed for the whole frame 
                                    but the tracking is from a video with samaller frame
    """

    check_file_exists(inverse_fisheye_map_file, raise_error=True)
    inverse_fisheye_maps = np.load(inverse_fisheye_map_file)

    # initialize coordinates
    coordinates = np.zeros((2, len(tracking['x'].values)))

    # extract coordinates
    for j, axis in enumerate(['x', 'y']):
        coordinates[j] = tracking[axis].values
        coordinates[j] = tracking[axis].values

    # convert original coordinates to registered coordinates
    coordinates[0] = inverse_fisheye_maps[
                                    coordinates[1].astype(np.uint16) + y_offset, 
                                    coordinates[0].astype(np.uint16) + x_offset, 0] - x_offset
    coordinates[1] = inverse_fisheye_maps[
                                    coordinates[1].astype(np.uint16) + y_offset, 
                                    coordinates[0].astype(np.uint16) + x_offset, 1] - y_offset

    # Return corrected coordinates
    corrected = tracking.copy()
    corrected['x'] = coordinates[0]
    corrected['y'] = coordinates[1]
    return corrected