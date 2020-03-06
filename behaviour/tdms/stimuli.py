
"""
    !!! THIS CODE WAS COPIED OVER FROM ANOTHER REPO BUT HAS NOT BEEN ADAPTED TO WORK HERE
"""

raise NotImplementedError

def extract_behaviour_stimuli(aifile):
    """extract_behaviour_stimuli [given the path to a .tdms file with session metadata extract
    stim names and timestamp (in frames)]
    
    Arguments:
        aifile {[str]} -- [path to .tdms file] 
    """
    # Get .tdms as a dataframe
    tdms_df, cols = open_temp_tdms_as_df(aifile, move=False)

    stim_cols = [c for c in cols if 'Stimulis' in c]
    stimuli = []
    stim = namedtuple('stim', 'type name frame')
    for c in stim_cols:
        stim_type = c.split(' Stimulis')[0][2:].lower()
        if 'digit' in stim_type: continue
        stim_name = c.split('-')[-1][:-2].lower()
        try:
            stim_frame = int(c.split("'/' ")[-1].split('-')[0])
        except:
            try:
                stim_frame = int(c.split("'/'")[-1].split('-')[0])
            except:
                continue
        stimuli.append(stim(stim_type, stim_name, stim_frame))
    return stimuli

def extract_ai_info(key, aifile):
    """
    aifile: str path to ai.tdms

    extract channels values from file and returns a key dict for dj table insertion

    """

    # Get .tdms as a dataframe
    tdms_df, cols = open_temp_tdms_as_df(aifile, move=True, skip_df=True)
    chs = ["/'OverviewCameraTrigger_AI'/'0'", "/'ThreatCameraTrigger_AI'/'0'", "/'AudioIRLED_AI'/'0'", "/'AudioFromSpeaker_AI'/'0'"]
    """ 
    Now extracting the data directly from the .tdms without conversion to df
    """
    key['overview_camera_triggers'] = np.round(tdms_df.object('OverviewCameraTrigger_AI', '0').data, 2)
    key['threat_camera_triggers'] = np.round(tdms_df.object('ThreatCameraTrigger_AI', '0').data, 2)
    key['audio_irled'] = np.round(tdms_df.object('AudioIRLED_AI', '0').data, 2)
    if 'AudioFromSpeaker_AI' in tdms_df.groups():
        key['audio_signal'] = np.round(tdms_df.object('AudioFromSpeaker_AI', '0').data, 2)
    else:
        key['audio_signal'] = -1
    key['ldr'] = -1  # ? insert here
    key['tstart'] = -1
    key['manuals_names'] = -1
    # warnings.warn('List of strings not currently supported, cant insert manuals names')
    key['manuals_timestamps'] = -1 #  np.array(times)
    return key