import matplotlib.pyplot as plt


class Axes(object):

    def __init__(self, ax, **kwargs):
        self.ax = ax
        self.fig = None
        self._title = None
        self._ylabel = None
        self._xlabel = None
        self._ylim = None
        self._xticklabels = None
        self._xticklabels_attrs = None
        self._xlabel_attrs = None
        self._ylabel_attrs = None
        self._facecolor = None
        self.update_axes_settings(**kwargs)

    def update_axes_settings(self, **kwargs):
        all_properties = {}
        for key, value in kwargs.items():
            all_properties['_%s' % key] = value
        self.__dict__.update(all_properties)

    @property
    def facecolor(self):
        return self._facecolor

    @facecolor.setter
    def facecolor(self, facecolor):
        self._facecolor = facecolor

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

    def calculate_and_show_percentage_or_number(self):
        # create a list to collect the plt.patches data
        totals = []
        # find the values and append to list
        for i in self.ax.patches:
            totals.append(i.get_height())
        total = sum(totals)
        for p in self.ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height() / total)
            x = p.get_x() + p.get_width()/2 - 0.02
            y = p.get_y() + p.get_height() + 0.06
            value = p.get_height()
            text = str(value) if getattr(self, '_display_number', False) else percentage
            self.ax.annotate(text, (x, y))

    def init_default_settings(self):
        self.ax.set_axisbelow(True)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.grid(color='white', linewidth='1.0')

    def apply_settings(self):
        attrs = [a for a in dir(self) if not a.startswith('__') and not a.startswith('_')]
        self.init_default_settings()
        for attr in attrs:
            attr_value = getattr(self, attr, None)
            if isinstance(attr_value, (float, str, dict, int, list)) and attr_value:
                property_attrs = getattr(self, '%s_attrs' % attr, {}) or {}
                func = getattr(self.ax, 'set_%s' % attr, None)
                if func:
                    func(attr_value, **property_attrs)

        if getattr(self, '_show_percentage', False) or getattr(self, '_display_number', False):
            self.calculate_and_show_percentage_or_number()
