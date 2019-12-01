import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager

class BaseObject(object):

    def __init__(self, figsize, **kwargs):
        self.ax = None
        self.fig = None
        self.figsize = figsize
        self._color = ['#009ACD', '#ADD8E6', '#63D1F4', '#0EBFE9',
                      '#C1F0F6', '#0099CC']
        self.__dict__.update(kwargs)
        self.init_default_settings()

    def init_default_settings(self):
        self.fig, self.ax = plt.subplots(figsize=(self.figsize or (12, 6)))
        plt.rcParams['font.sans-serif'] = getattr(self, 'font', None) or 'Arial'
        plt.rcParams['font.family'] = getattr(self, 'font_family',
                                              None) or 'sans-serif'
        plt.rcParams['text.color'] = getattr(self, 'text_color',
                                             None) or '#909090'
        plt.rcParams['axes.labelcolor'] = getattr(self, 'axes_label_color',
                                                  None) or '#909090'
        plt.rcParams['xtick.color'] = getattr(self, 'xtick_color',
                                              None) or '#909090'
        plt.rcParams['ytick.color'] = getattr(self, 'ytick_color',
                                              None) or '#909090'
        plt.rcParams['font.size'] = getattr(self, 'font_size', None) or 12