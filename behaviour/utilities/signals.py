import numpy as np


def convert_from_sample_to_frame(sample_n, sampling_rate, fps):
    '''
        Given the time at which an event occurred in # samples, get the 
        corresponding frame number assuming that the sampling rate and the frame
        rate of the recording are different but recording of both
        samples and frames started at the same time.
    '''

    return np.round(np.multiply(np.divide(sample_n, sampling_rate), fps))


def get_frames_times_from_squarewave_signal(squarewave_signal, th=4,):
    if squarewave_signal[0]>th and squarewave_signal[1]>th:
        squarewave_signal[0] = 0
    derivative = np.concatenate([[0], np.diff(squarewave_signal)])
    return np.where(derivative > th)[0]


def get_times_signal_high_and_low(signal, th=1, min_time_between_highs=None):
    """
        Given a 1d time series it returns the times 
        (in # samples) in which the signal goes low->high (onset)
        and high->low (offset)

        :param signal: 1d numpy array or list with time series data
        :param th: float, the time derivative of signal is thresholded to find onset and offset
        :param min_time_between_highs: int, if a number of passes, onsets that are closer in time
                than this value are ignored.
    """
    signal_copy = np.zeros_like(signal)
    signal_copy[signal > th] = 1

    signal_onset = np.where(np.diff(signal_copy) > .5)[0]
    if min_time_between_highs is not None:
        signal_onset = np.concatenate([[signal_onset[0]], 
                            signal_onset[np.where(np.diff(signal_onset)>min_time_between_highs)[0]+1]])


    signal_offset = np.where(np.diff(signal_copy) < -.5)[0]
    if min_time_between_highs is not None:
        signal_offset = np.concatenate([[signal_offset[0]], 
                            signal_offset[np.where(np.diff(signal_offset)>min_time_between_highs)[0]+1]])
    return signal_onset, signal_offset
