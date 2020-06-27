from data_visualisation.figure import Figure
import seaborn as sns
import numpy as np
from utils.utils import merge_nested_dicts


class Plot(object):

    def __init__(self, data):
        self._data = data
        self._figure_attrs = None

    @staticmethod
    def init_figure_and_update_axes_settings(kwargs):
        figure_size = kwargs.get('figure_size', (10, 10))
        axes_attrs = kwargs.get('axes_settings', {})
        kwargs.pop('figure_size', None)
        kwargs.pop('axes_settings', None)
        f = Figure(figure_size)
        f.axes[0].update_axes_settings(**axes_attrs)
        return f.axes[0]

    @staticmethod
    def get_axis_and_settings(**kwargs):
        axis = kwargs.pop('axes', None)
        kwargs = dict(merge_nested_dicts(kwargs, {'axes_settings': {'facecolor': '#E5E5E5'}}))
        if axis:
            axis.update_axes_settings(**kwargs.get('axes_settings'))
        else:
            axis = Plot.init_figure_and_update_axes_settings(kwargs)
        kwargs.pop('axes_settings', None)
        return axis, kwargs

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def plot(self, plot_type, x_key, y_key, **kwargs):
        axis, kwargs = Plot.get_axis_and_settings(**kwargs)
        func = getattr(sns, plot_type, None)
        if func:
            func(x=x_key, y=y_key, data=self.data, ax=axis.ax, **kwargs)
        axis.apply_settings()

    def scatter(self, x_key, y_key, **kwargs):
        self.plot('scatterplot', x_key, y_key, **kwargs)

    def bar(self, x_key, y_key, **kwargs):
        self.plot('barplot', x_key, y_key, **kwargs)

    def line(self, x_key, y_key, **kwargs):
        self.plot('lineplot', x_key, y_key, **kwargs)

    def count(self, x_key, **kwargs):
        axis, kwargs = Plot.get_axis_and_settings(**kwargs)
        sns.countplot(x=x_key, data=self.data, ax=axis.ax, **kwargs)
        axis.apply_settings()

    def plot_bulk(self, all_plots, **extra_args):
        single_figure = extra_args.get('single_figure', None)
        single_axes = extra_args.get('single_axes', False)
        all_plots = np.array(all_plots)
        figure = getattr(self, 'figure', None)
        if not figure:
            figsize = extra_args.get('figsize', (10, 10))
            if single_axes:
                nrows = 1
                ncols = 1
            else:
                nrows, ncols = all_plots.shape
            figure = Figure(figsize, nrows=nrows, ncols=ncols)
            if single_figure:
                self.figure = figure
        i = 0
        plots = all_plots.flatten()
        for plot in plots:
            plot_func = getattr(self, plot['type'], None)
            if plot_func:
                x_key, y_key, kwargs = plot['params']
                kwargs['axes'] = figure.axes[i]
                plot_func(x_key, y_key, **kwargs)
            else:
                raise Exception("Not a valid plot func %s " % plot['type'])
            if not single_axes:
                i = i + 1
