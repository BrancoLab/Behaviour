import cv2
import numpy as np
import scipy.misc
from termcolor import colored
from tqdm import tqdm
import glob
import os

from fcutils.file_io.utils import check_file_exists, listdir


from behaviour.utilities.video import get_background_from_video


def register_tracking_data(unregistered, M):
    """[Corrects tracking data (as extracted by DLC) using a transform Matrix obtained via the CommonCoordinateBehaviour
        toolbox. ]

    Arguments:
        unregistered {[np.ndarray]} -- [n-by-1 array]
        M {[np.ndarray]} -- [2-by-3 transformation matrix: https://github.com/BrancoLab/Common-Coordinate-Behaviour]

    Returns:
        registered {[np.ndarray]} -- [n-by-3 array with registered X,Y tracking and Velocity data]
    """     
    # Prep vars
    m3d = np.append(M, np.zeros((1,3)),0)

    n_samples = unregistered.shape[0]
    unregistered = np.repeat(unregistered, 2).reshape(n_samples, 2)
    registered = np.zeros_like(unregistered)

    # affine transform to match model arena
    concat = np.ones((len(unregistered), 3))
    concat[:, :2] = unregistered
    registered = np.matmul(m3d, concat.T).T[:, :2]
    return registered




class CommonCoordinates:
    def __init__(self, arena_func=None, arena_image=None, points=None):
        """
            Initialize the CommonCoordinates class used to register video frames 
            to a user given arena template.

            :param arena_func: function used to generate an immage (np.array) of arena template
            :param arena_image: either path to a saved template image or an image (np.array)
            :param points: list of two-tuples with coordinates of points to use for registration
                            need at least 5 points.
        """
        # ---------------------------- LOAD ARENA TEMPLATE --------------------------- #
        if arena_func is not None:
            self.arena = arena_func()
        elif arena_image is not None:
            if isinstance(arena_image, str):
                try:
                    self.arena = cv2.imread(arena_image)  
                except Exception as e:
                    raise FileNotFoundError("Could not load arena from path  {}\n{}".format(arena_image, e))
            else:
                self.arena = arena_image.copy()
                if len(self.arena.shape) == 2: # make sure it's a RGB image
                    self.arena = np.repeat(self.arena[:, :, np.newaxis], 3, axis=2)
        else: 
            raise ValueError("Either arena func or arena image need to be specified")

        # Get points
        if len(points) < 5:
            raise ValueError("Sorry, list of points used for registration too short. Need at least 5 points")
        else:
            self.points = points


    def get_video_transform_mtx(self, videopath, background=None,
                output_dir=None, correct_fisheye=False, 
                fisheye_kwargs={}, overwrite=False):

        # ---------------------------------- Set up ---------------------------------- #
        # Check video is good
        check_file_exists(videopath, raise_error=True)

        # Get output directory and save name and check if it exists
        if output_dir is None:
            output_dir = os.path.split(videopath)[0]
        video_name = os.path.split(videopath)[-1].split(".")[0]
        save_path = os.path.join(output_dir, video_name+"_transform_mtx.npy")

        if save_path in list_dir(output_dir) and not overwrite:
            print("A transform matrix already exists, loading it")
            return np.load(save_path)

        # Get background
        if background is None:
            background = get_background(videopath)

        # Check if we need to apply fisheye correction
        if correct_fisheye:
            raise NotImplementedError("Sorry, fisheye correction not ready")
            maps = np.load(fisheye_map_location)
            map1 = maps[:, :, 0:2]
            map2 = maps[:, :, 2]*0

            bg_copy = cv2.copyMakeBorder(background, y_offset, int((map1.shape[0] - background.shape[0]) - y_offset),
                                                    x_offset, int((map1.shape[1] - background.shape[1]) - x_offset), cv2.BORDER_CONSTANT, value=0)

            bg_copy = cv2.remap(bg_copy, map1, map2, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            bg_copy = bg_copy[y_offset:-int((map1.shape[0] - background.shape[0]) - y_offset),
                                x_offset:-int((map1.shape[1] - background.shape[1]) - x_offset)]
        else:
            bg_copy = background.copy()


        # ------------------------------ Initialize GUI ------------------------------ #
        # initialize clicked point arrays
        background_data = dict(bg_copy=bg_copy, clicked_points=np.array(([], [])).T) # [background_copy, np.array(([], [])).T]
        arena_data = dict(temp=[], points=[])  # [[], np.array(([], [])).T]

        # add 1-2-3-4 markers to model arena to show where points need to go
        for i, point in enumerate(self.points.astype(np.uint32)):
            self.arena = cv2.circle(self.arena, (point[0], point[1]), 3, 255, -1)
            self.arena = cv2.circle(self.arena, (point[0], point[1]), 4, 0, 1)
            cv2.putText(self.arena, str(i+1), tuple(point), 0, .55, 150, thickness=2)

            point = np.reshape(point, (1, 2))
            arena_data['points'] = np.concatenate((arena_data['points'], point)) # Append point to arena data

        # initialize windows
        cv2.startWindowThread()
        cv2.namedWindow('background')
        cv2.imshow('background', bg_copy)
        cv2.namedWindow('model arena')
        cv2.imshow('model arena', arena)

        # create functions to react to clicked points
        cv2.setMouseCallback('background', self.select_transform_points, background_data)  # Mouse callback

        
        # --------------------------- User interaction loop -------------------------- #
        # take in clicked points until all points are clicked or user presses 'q'
        while True: 
            cv2.imshow('background', bg_copy)

            number_clicked_points = background_data['clicked_points'].shape[0]
            if number_clicked_points == len(self.points):
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # perform projective transform and register background
        M = cv2.estimateRigidTransform(background_data['clicked_points'], arena_data['points'], False)
        registered_background = cv2.warpAffine(bg_copy, M, background.shape[::-1])

        # Start user interaction to refine the matrix
        M = self.get_mtrx_user_interaction()

        return M


    # ---------------------------------------------------------------------------- #
    #                               USER INTERACTION                               #
    # ---------------------------------------------------------------------------- #
    def get_mtrx_user_interaction(self, arena, background, registered_background,           
                                    background_data, arena_data, M, bg_copy,
                                    alpha=.7, colors = [[150, 0, 150], [0, 255, 0]]):
        color_array = make_color_array(colors, background.shape)

        registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                 * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
        arena_color = (cv2.cvtColor(arena.copy(), cv2.COLOR_GRAY2RGB)
                       * np.squeeze(color_array[:, :, :, 1])).astype(np.uint8)

        overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
        cv2.imshow('registered background', overlaid_arenas)

        # --------------------------------------------------
        # initialize GUI for correcting transform
        # --------------------------------------------------
        print("\n  On the 'registered background' pane: Left click model arena --> Right click model background")
        print('  (Model arena is green at bright locations and purple at dark locations)')
        print('\n  Advanced users: use arrow keys and \'wasd\' to fine-tune translation and scale as a final step')
        print('  Crème de la crème: use \'tfgh\' to fine-tune shear\n')
        print('  y: save and use transform')
        print('  r: reset transform')
        update_transform_data = dict(clicked_points=overlaid_arenas,  # background_data['clicked_points'], 
                        points=arena_data['points'], M=M, bg_copy=bg_copy)

        # create functions to react to additional clicked points
        cv2.setMouseCallback('registered background', self.additional_transform_points, update_transform_data)

        # take in clicked points until 'q' is pressed
        initial_number_clicked_points = [update_transform_data['points'].shape[0], update_transform_data['M'].shape[0]]
        M_initial = M
        M_indices = [(0,2),(1,2),(0,0),(1,1),(0,1),(1,0),(2,0),(2,2)]
        # M_indices_meanings = ['x-translate','y-translate','x-scale','y-scale','x->y shear','y->x shear','x perspective','y perspective']
        M_mod_keys = [2424832, 2555904, 2490368, 2621440, ord('w'), ord('a'), ord('s'), ord('d'), ord('f'), ord('t'),
                      ord('g'), ord('h'), ord('j'), ord('i'), ord('k'), ord('l')]
        while True:
            cv2.imshow('registered background',overlaid_arenas)
            cv2.imshow('background', registered_background)
            number_clicked_points = [update_transform_data['points'].shape[0], update_transform_data['M'].shape[0]]
            update_transform = False
            k = cv2.waitKey(10)
            # If a left and right point are clicked:
            if number_clicked_points[0]>initial_number_clicked_points[0] and number_clicked_points[1]>initial_number_clicked_points[1]:
                initial_number_clicked_points = number_clicked_points
                # update transform and overlay images
                try:
                    M = cv2.estimateRigidTransform(update_transform_data['points'], update_transform_data['M'],False) #True ~ full transform
                    update_transform = True
                except:
                    continue
            elif k in M_mod_keys: # if an arrow key if pressed
                if k == 2424832: # left arrow - x translate
                    M[M_indices[0]] = M[M_indices[0]] - abs(M_initial[M_indices[0]]) * .005
                elif k == 2555904: # right arrow - x translate
                    M[M_indices[0]] = M[M_indices[0]] + abs(M_initial[M_indices[0]]) * .005
                elif k == 2490368: # up arrow - y translate
                    M[M_indices[1]] = M[M_indices[1]] - abs(M_initial[M_indices[1]]) * .005
                elif k == 2621440: # down arrow - y translate
                    M[M_indices[1]] = M[M_indices[1]] + abs(M_initial[M_indices[1]]) * .005
                elif k == ord('a'): # down arrow - x scale
                    M[M_indices[2]] = M[M_indices[2]] + abs(M_initial[M_indices[2]]) * .005
                elif k == ord('d'): # down arrow - x scale
                    M[M_indices[2]] = M[M_indices[2]] - abs(M_initial[M_indices[2]]) * .005
                elif k == ord('s'): # down arrow - y scale
                    M[M_indices[3]] = M[M_indices[3]] + abs(M_initial[M_indices[3]]) * .005
                elif k == ord('w'): # down arrow - y scale
                    M[M_indices[3]] = M[M_indices[3]] - abs(M_initial[M_indices[3]]) * .005
                elif k == ord('f'): # down arrow - x-y shear
                    M[M_indices[4]] = M[M_indices[4]] - abs(M_initial[M_indices[4]]) * .005
                elif k == ord('h'): # down arrow - x-y shear
                    M[M_indices[4]] = M[M_indices[4]] + abs(M_initial[M_indices[4]]) * .005
                elif k == ord('t'): # down arrow - y-x shear
                    M[M_indices[5]] = M[M_indices[5]] - abs(M_initial[M_indices[5]]) * .005
                elif k == ord('g'): # down arrow - y-x shear
                    M[M_indices[5]] = M[M_indices[5]] + abs(M_initial[M_indices[5]]) * .005

                update_transform = True

            elif k == ord('q') or k == ord('y'):
                print(colored('\nRegistration completed\n', 'green'))
                break

            if update_transform:
                update_transform_data['M'] = M
                # registered_background = cv2.warpPerspective(background_copy, M, background.shape)
                registered_background = cv2.warpAffine(background_copy, M, background.shape[::-1])
                registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                               * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
                overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
                update_transform_data['points'] = overlaid_arenas

        cv2.destroyAllWindows()


        return M


    # ---------------------------------------------------------------------------- #
    #                                CALLBACK FUNCS and utils                      #
    # ---------------------------------------------------------------------------- #
    @staticmethod
    def select_transform_points(event,x,y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:

            data[0] = cv2.circle(data[0], (x, y), 3, 255, -1)
            data[0] = cv2.circle(data[0], (x, y), 4, 0, 1)

            clicks = np.reshape(np.array([x, y]),(1,2))
            data[1] = np.concatenate((data[1], clicks))

    @staticmethod
    def additional_transform_points(event,x,y, flags, data):
        if event == cv2.EVENT_RBUTTONDOWN:

            data[0] = cv2.circle(data[0], (x, y), 2, (200,0,0), -1)
            data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

            M_inverse = cv2.invertAffineTransform(data[3])
            transformed_clicks = np.matmul(np.append(M_inverse,np.zeros((1,3)),0), [x, y, 1])


            data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 2, (0, 0, 200), -1)
            data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 3, 0, 1)

            clicks = np.reshape(transformed_clicks[0:2],(1,2))
            data[1] = np.concatenate((data[1], clicks))
        elif event == cv2.EVENT_LBUTTONDOWN:

            data[0] = cv2.circle(data[0], (x, y), 2, (0,200,200), -1)
            data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

            clicks = np.reshape(np.array([x, y]),(1,2))
            data[2] = np.concatenate((data[2], clicks))


    @staticmethod
    def make_color_array(colors, image_size):
        color_array = np.zeros((image_size[0],image_size[1], 3, len(colors)))  # create coloring arrays
        for c in range(len(colors)):
            for i in range(3):  # B, G, R
                color_array[:, :, i, c] = np.ones((image_size[0],image_size[1])) * colors[c][i] / sum(
                    colors[c])
        return color_array

