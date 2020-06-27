import numpy as np
from optimizers.gradient_descent import GradientDescent


class LinearRegressor(object):

    def __init__(self, X, Y, **kwargs):
        self.X = X
        self.Y = Y
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.iterations = kwargs.get('iterations', 1000)
        self.batch_size = kwargs.get('batch_size', 20)
        self.cost_history = None
        self.w = None

    def fit(self):
        gd = GradientDescent(self.learning_rate, self.X, self.Y)
        gd.iterations = self.iterations
        gd.theta = np.zeros((self.shape[1], 1))
        gd.optimise_mini_batch_gd(batch_size=self.batch_size)
        self.cost_history = gd.cost_history
        self.w = gd.theta

    def predict(self):
        return np.dot(self.X, self.w)
