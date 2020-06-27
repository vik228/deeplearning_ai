import matplotlib.pyplot as plt
from data_visualisation.axes import Axes


class Figure(object):

    @staticmethod
    def resolve_axes(ax, nrows, ncols):
        axes = []
        if nrows*ncols == 1:
            return [Axes(ax)]
        else:
            for ax in ax.flatten():
                axes.append(Axes(ax))
        return axes

    def __init__(self, figsize=(10, 10), **kwargs):
        nrows = kwargs.get('nrows', 1)
        ncols = kwargs.get('ncols', 1)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        self.fig = fig
        self.axes = Figure.resolve_axes(ax, nrows, ncols)
        self.plt = plt

