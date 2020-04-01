# Behaviour
Collection of shared utilities to process and analyse behaviour data


# Installation
Install with
```
pip install git+https://github.com/BrancoLab/Behaviour.git
```

Update with:
```
pip install git+https://github.com/BrancoLab/Behaviour.git --upgrade
```

The function in this repository require [fc utils](https://github.com/FedeClaudi/fcutils), please 
install/upgrade it with:
```
pip install git+https://github.com/FedeClaudi/fcutils.git --upgrade
```

# USAGE
## TDMS
Collection of functions useful for handling .tdms files from mantis. 

### utils.py
Functions to open, parse and save .tdms files

### mantis_videoframes_test.py
Functions to check if frames where dropped in a mantis recording (using video and metadata files).



## TRACKING
Functions for post processing and handling of DLC output

### roi_stats.py
Code to use DLC tracking to check time spent in user defined ROIs

### tracking.py
Post processing of DLC tracking data: median filtering, computation of kinematics, body-segments analysis etc

## SIGNALS [utilities > signals.py]
Code to analyse 1d time series to extract relevant events (e.g. stimuli onsets from a LDR trace)

## VIDEOS
### utils.py
Lot's of useful stuff for handling, loading and saving videos with opencv. 

### trial_videos.py
Code to create a video with short clips for each trial within an experimental session

