import numpy as np

def get_frames_times_from_squarewave_signal(squarewave_signal, th=4,):
    if squarewave_signal[0]>th and squarewave_signal[1]>th:
        squarewave_signal[0] = 0
    derivative = np.concatenate([[0], np.diff(squarewave_signal)])
    return np.where(derivative > th)[0]


def get_times_signal_high_and_low(signal, th=1):
    """
        Given a 1d time series it returns the times 
        (in # samples) in which the signal goes low->high (onset)
        and high->low (offset)

        :param signal: 1d numpy array or list with time series data
        :param th: float, the time derivative of signal is thresholded to find onset and offset
    """
    signal_copy = np.zeros_like(signal)
    signal_copy[signal > th] = 1

    signal_onset = np.where(np.diff(signal_copy) > .5)[0]
    signal_offset = np.where(np.diff(signal_copy) < -.5)[0]
    return signal_onset, signal_offset
