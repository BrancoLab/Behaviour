import pandas as pd
import numpy as np
import os


def get_scorer_bodyparts(tracking):
    """
        Given the tracking data hierarchical df from DLC, return
        the scorer and bodyparts names
    """
    first_frame = tracking.iloc[0]
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
    tracking = tracking.unstack()

    trackings = {}
    for bp in bodyparts:
        tr = {c:tracking.loc[scorer, bp, c].values for c in ['x', 'y', 'likelihood']}
        trackings[bp] = pd.DataFrame(tr)

    return trackings, bodyparts

