# Import packages
from Video_Functions import peri_stimulus_video_clip, register_arena, get_background
from termcolor import colored

'''
--------------   SET PARAMETERS    --------------
'''

# file path of behaviour video to register
video_file_path = 'C:\\Drive\\Common-Coordinate-Behaviour\\behavior videos\\CA3481_loom.mp4'

# file path of behaviour clip to save
save_file_path = 'C:\\Drive\\Common-Coordinate-Behaviour\\clips'

# file path of fisheye correction -- set to an invalid location such as '' to skip fisheye correction
# A corrective mapping for the Branco lab's typical camera is included in the repo!
fisheye_map_location = 'C:\\Drive\\Common-Coordinate-Behaviour\\fisheye calibration maps\\fisheye_maps.npy'

# frame of stimulus onset
stim_frame = 675

# seconds before and after stimulus to display
window_pre = 3
window_post = 7

# frames per second of video
fps = 30

# name of experiment
experiment = 'Circle wall up'

# name of animal
animal_id = 'CA3481'

# stimulus type
stim_type = 'visual'

# The fisheye correction works on the entire frame. If not recording full-frame, put the x and y offset here
x_offset = 120
y_offset = 300





'''
--------------   GET BACKGROUND IMAGE    --------------
'''
print(colored('\nFetching background', 'green'))
background_image = get_background(video_file_path, start_frame=1000, avg_over=10)



'''
--------------   REGISTER ARENA TO COMMON COORDINATE BEHAVIOUR    --------------
'''
print(colored('\nRegistering arena', 'green'))
# Important: you'll need to modify the model_arena function (1st function in Video_Functions) using opencv,
#            to reflect your own arena instead of ours
registration = register_arena(background_image, fisheye_map_location, x_offset, y_offset, show_arena = False)
# This outputs a variable called 'registration' which is used in the video clip function below to register each frame
# feel free to save it, using: np.save(experiment + animal +'_transform',registration)



'''
--------------   SAVE VIDEO CLIPS    --------------
'''
videoname = '{}_{}_{}-{}\''.format(experiment,animal_id,stim_type, round(stim_frame / fps / 60))
print(colored('Creating behavior clip for ' + videoname, 'green'))

peri_stimulus_video_clip(video_file_path, videoname, save_file_path, stim_frame, window_pre, window_post,
                         registration, x_offset, y_offset, fps=fps, save_clip = True, counter = True)

print(colored('done'))