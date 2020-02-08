import matplotlib.pyplot as plt

from fcutils.maths.stimuli_detection import find_peaks_in_signal


def get_frames_times_from_squarewave_signal(
    squarewave_signal, time_limit=10, th=4, debug=False
):
    frame_starts = find_peaks_in_signal(squarewave_signal, 10, 1, above=True)

    if debug:
        f, ax = plt.subplots()
        ax.plot(squarewave_signal)
        ax.scatter(frame_starts, [th for _ in frame_starts])
        plt.show()
    return frame_starts

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