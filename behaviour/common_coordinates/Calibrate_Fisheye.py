import numpy as np; import os; import glob; import cv2

'''
FISHEYE CALIBRATION AND CORRECTION CODE
meat of the code comes from: https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
see Kenneth Jiang's blog post above for helpful explanation
'''


'''
SET PARAMETERS
'''
# Select calibration images folder location
# the checkerboard should be large and very clearly visible and from as many orientations (in 3D) as possible
calibration_images_loc = 'C:\\Drive\\Common-Coordinate-Behaviour\\example fisheye calibration images\\'
image_extension = '.png'

# Name prepended to the saved rectification maps
camera = 'upstairs'

# Set parameters
CHECKERBOARD = (28,12) #size of the checkerboard (# of vertices in each dimension, not including those on the edge)

# The algorithm is finnicky and likes saturation, so set pixels with values below the dark_threshold to black
dark_threshold = 30




'''
FIND THE CHECKERBOARD CORNERS IN THE IMAGES
(pretty slow)
'''
# Find checkerboard corners -- set up for .pngs
CHECKERFLIP = tuple(np.flip(CHECKERBOARD,0))
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW #+cv2.fisheye.CALIB_CHECK_COND
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob(calibration_images_loc + '*' + image_extension) # find the images in the folder
print('found ' + str(len(images)) + ' images for calibration.')

for fname in images:
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    
    calib_image_pre = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    #increase contrast
    # for light_threshold in light_thresholds:
    calib_image = calib_image_pre

    calib_image[calib_image<dark_threshold] = 0
    # calib_image[calib_image>light_threshold] = 255

    cv2.imshow('calibration image',calib_image)
    cv2.waitKey(5)

    # Find the chess board corners (takes a while)
    ret, corners = cv2.findChessboardCorners(calib_image, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print(fname + ': successfully identified corners')

        # objpoints = np.append(objpoints, objp)
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(calib_image,corners,(11,11),(-1,-1),subpix_criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(calib_image, CHECKERBOARD, corners2, ret)
        cv2.imshow('calibration image', calib_image)
        cv2.waitKey(500)
    else:
        print(fname + ': failed to identify corners')
        
        
        
'''
USE THE CORNERS FOUND ABOVE TO GET THE CALIBRATION MATRICES K AND D
'''
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



'''
TEST THE CALIBRATION AND SAVE THE REMAPPINGS
'''
# Display recalibrated images
DIM=_img_shape[::-1]
K=np.array(K)
D=np.array(D)
for img_path in images:
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


'''
SAVE MAPS TO USE IN ANALYSIS
see the register_frame() function in Video_Functions.py for an example of how to use this for frame-by-frame correction
'''
maps = np.zeros((calib_image.shape[0],calib_image.shape[1],3)).astype(np.int16)
maps[:,:,0:2] = map1
maps[:,:,2] = map2
np.save(calibration_images_loc + 'fisheye_maps_' + camera + '.npy', maps)




        