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
