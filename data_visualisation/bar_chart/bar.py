from .base_object import BaseObject


class Bar(BaseObject):

    def __init__(self, figsize, **kwargs):
        self.__dict__.update(kwargs)
        self._bar_heights = None
        self._x_coords = None
        self._title = None
        self._ylabel = None
        self._xlabel = None
        self._ylim = None
        self._xticklabels = None
        self._xticklabels_attrs = None
        self._xlabel_attrs = None
        self._ylabel_attrs = None
        self.ignore_list = ['x_coords', 'bar_heights', 'color', 'init_default_settings', 'ignore_list', 'ax', 'fig',
                            'figsize', 'show', 'xticklabels_attrs', 'show_percentage', 'xlabel_attrs', 'ylabel_attrs']
        BaseObject.__init__(self, figsize)

    @property
    def bar_heights(self):
        return self._bar_heights

    @bar_heights.setter
    def bar_heights(self, bar_heights):
        self._bar_heights = bar_heights

    @property
    def x_coords(self):
        return self._x_coords

    @x_coords.setter
    def x_coords(self, x_coords):
        self._x_coords = x_coords

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title

    @property
    def ylabel(self):
        return self._ylabel

    @ylabel.setter
    def ylabel(self, label):
        self._ylabel = label

    @property
    def xlabel(self):
        return self._xlabel

    @xlabel.setter
    def xlabel(self, label):
        self._xlabel = label

    @property
    def ylim(self):
        return self._ylim

    @ylim.setter
    def ylim(self, lim):
        self._ylim = lim

    @property
    def xticklabels(self):
        return self._xticklabels

    @xticklabels.setter
    def xticklabels(self, xticklabels):
        self._xticklabels = xticklabels

    @property
    def xticklabels_attrs(self):
        return self._xticklabels_attrs

    @xticklabels_attrs.setter
    def xticklabels_attrs(self, attr_dict):
        self._xticklabels_attrs = attr_dict

    @property
    def xlabel_attrs(self):
        return self._xlabel_attrs

    @xlabel_attrs.setter
    def xlabel_attrs(self, attr_dict):
        self._xlabel_attrs = attr_dict

    @property
    def ylabel_attrs(self):
        return self._ylabel_attrs

    @ylabel_attrs.setter
    def ylabel_attrs(self, attr_dict):
        self.ylabel_attrs = attr_dict

    def show_percentage(self, horizontal=False):
        # create a list to collect the plt.patches data
        totals = []

        # find the values and append to list
        for i in self.ax.patches:
            totals.append(i.get_height())
        total = sum(totals)
        for i in self.ax.patches:
            if horizontal:
                x = i.get_x() + i.get_width()
                y = i.get_y() + i.get_height() / 2
            else:
                x = i.get_x() + i.get_width() / 2
                y = i.get_y() + i.get_height()
            self.ax.text(x, y, str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=15,
                         color='dimgrey', ha='center', va='bottom')

    def show(self, **kwargs):
        show_percentage = kwargs.get('show_percentage', False)
        horizontal = kwargs.get('horizontal', False)
        if horizontal:
            self.ax.barh(self.x_coords, self.bar_heights, color=self.color)
        else:
            self.ax.bar(self.x_coords, self.bar_heights, color=self.color)
        attrs = [a for a in dir(self) if not a.startswith('__') and not a.startswith('_')]
        for attr in attrs:
            if attr not in self.ignore_list:
                attr_value = getattr(self, attr)
                if attr_value:
                    property_attrs = getattr(self, '%s_attrs' % attr, {})
                    getattr(self.ax, 'set_%s' % attr)(attr_value, **property_attrs)
        if show_percentage:
            self.show_percentage(horizontal=horizontal)
