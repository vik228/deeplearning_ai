from data_visualisation.axes import BaseObject


class Bar(BaseObject):
    def __init__(self, figsize, **kwargs):
        self.__dict__.update(kwargs)
        self._bar_heights = None
        self._x_coords = None
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

    def show_percentage(self, horizontal=False, show_percentage=False):
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
                value1 = i.get_width()
                value2 = str(round((i.get_width() / total) * 100, 2)) + "%"
            else:
                x = i.get_x() + i.get_width() / 2
                y = i.get_y() + i.get_height()
                value1 = i.get_height()
                value2 = str(round((i.get_height() / total) * 100, 2)) + "%"
            value = str(value1) if not show_percentage else str(value2) + "%"
            self.ax.text(
                x, y, value, fontsize=15, color="dimgrey", ha="center", va="bottom"
            )

    def show(self, **kwargs):
        show_percentage = kwargs.get("show_percentage", False)
        horizontal = kwargs.get("horizontal", False)
        if horizontal:
            self.ax.barh(self.x_coords, self.bar_heights, color=self.color)
        else:
            self.ax.bar(self.x_coords, self.bar_heights, color=self.color)
        self.apply_settings()
        self.show_percentage(horizontal=horizontal, show_percentage=show_percentage)
