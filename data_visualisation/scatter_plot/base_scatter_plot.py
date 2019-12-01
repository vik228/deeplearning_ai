import matplotlib.pyplot as plt

class BaseScatterPlot(object):

    def __init__(self, figsize, **kwargs):
        self.figsize = figsize
        self._x = None
        self._y = None
        self._label = None
        self.__dict__.update(kwargs)
        self.plt = plt

    @property
    def x_values(self):
        return self._x

    @x_values.setter
    def x_values(self, x):
        self._x = x

    @property
    def y_values(self):
        return self._y

    @y_values.setter
    def y_values(self, y):
        self._y = y

    @property
    def label(self):
        return self.label

    @label.setter
    def label(self, label):
        self._label = label
