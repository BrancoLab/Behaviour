import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_tracking_args(tracking=None, x=None, y=None):
    """
        Handles tracking inputs passed in various ways

        :param tracking: mxN or Nxm np.ndarray with X,Y for each frame.
                Assumes that X and Y are the first two rows/columns in the 
                dimension with the fewest emelements.
        :param x: 1d np.ndarray or list with x coordintes
        :param y: 1d np.ndarray or list with y coordintes
    """

    if tracking is not None:
        if tracking.shape[0]<tracking.shape[1]: 
            return tracking[0, :], tracking[1, :]
        else:
            return tracking[:, 0], tracking[:, 1]
    else:
        if x is None or y is None:
            raise ValueError("Pass either tracking or x and y. Not enough arguments passed.")
        else:
            return x, y


def parse_figure_args(ax=None, **kwargs):
    if ax is None:
        f, ax = plt.subplots()
    ax.set(**kwargs)
    return ax



def plot_tracking_2d_trace(tracking=None, x=None, y=None, ax=None, 
                                line_kwargs={},  ax_kwargs={}):
    """
        Plots 2D tracking data as a line

        :param tracking: mxN or Nxm np.ndarray with X,Y for each frame.
                Assumes that X and Y are the first two rows/columns in the 
                dimension with the fewest emelements.
        :param x: 1d np.ndarray or list with x coordintes
        :param y: 1d np.ndarray or list with y coordintes
        :param ax: matplotlib ax, if not passed a figure created
        :param line_kwargs: dict with arguments to personalise how the line looks
        :param ax_kwargs: dict with extra arguments to personalise ax
    """
    x,y = parse_tracking_args(tracking=tracking, x=x, y=y)
    ax = parse_figure_args(ax=ax, **ax_kwargs)

    ax.plot(x, y, **line_kwargs)
    return ax



def plot_tracking_2d_heatmap(tracking=None, x=None, y=None, ax=None, 
                                cmap='Reds', kind='hex',
                                plot_kwargs={}, ax_kwargs={}):
    """
        Plots a 2D heatmap from tracking data. 

        :param tracking: mxN or Nxm np.ndarray with X,Y for each frame.
                Assumes that X and Y are the first two rows/columns in the 
                dimension with the fewest emelements.
        :param x: 1d np.ndarray or list with x coordintes
        :param y: 1d np.ndarray or list with y coordintes
        :param ax: matplotlib ax, if not passed a figure created
        :param cmap: colormap used
        :param kind: str, type of 2d histogram, either hex or hist
        :param plot_kwargs: dict with extra arguments to personalise plot
        :param ax_kwargs: dict with extra arguments to personalise ax
    """
    x,y = parse_tracking_args(tracking=tracking, x=x, y=y)
    ax = parse_figure_args(ax=ax, **ax_kwargs)

    if kind.lower() == 'hex':
        ax.hexbin(x, y, cmap=cmap, **plot_kwargs)
    elif kind.lower() == 'hist':
        ax.hist2d(x, y, cmap=cmap, **plot_kwargs)
    else:
        raise ValueError("Unrecognized histogram type")

    return ax
    




