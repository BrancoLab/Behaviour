import numpy as np

def get_frames_times_from_squarewave_signal(squarewave_signal, th=4,):
    if squarewave_signal[0]>th and squarewave_signal[1]>th:
        squarewave_signal[0] = 0

    derivative = np.concatenate([[0], np.diff(squarewave_signal)])
    return np.where(derivative > th)[0]