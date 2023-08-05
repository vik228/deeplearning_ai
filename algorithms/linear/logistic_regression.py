import numpy as np
from utils.activation_functions import sigmoid
from optimizers.gradient_descent import GradientDescent


class LogisticRegressor(object):
    def __init__(self, eta=0.00000001, n_iterations=1000):
        self.eta = eta
        self.n_iterations = n_iterations
        self.w = []

    @staticmethod
    def probability(x, theta):
        return sigmoid(LogisticRegressor.net_input(x, theta))

    @staticmethod
    def cost_function(theta, x, y):
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(
            y * np.log(LogisticRegressor.probability(x, theta))
            + (1 - y) * np.log(1 - LogisticRegressor.probability(x, theta))
        )
        return total_cost

    @staticmethod
    def net_input(x, theta):
        return np.dot(x, theta)

    @staticmethod
    def gradient(theta, x, y):
        m = x.shape[0]
        return (1 / m) * np.dot(
            x.T, (sigmoid(LogisticRegressor.net_input(x, theta)) - y)
        )

    def fit(self, x, y):
        self.w = np.zeros((x.shape[1], 1))
        gd = GradientDescent(self.eta, x, y)
        gd.iterations = self.n_iterations
        gd.theta = self.w
        gd.optimise_mini_batch_gd(
            func=LogisticRegressor.cost_function, gradient=LogisticRegressor.gradient
        )
        self.w = gd.theta

    def accuracy(self, x, actual_classes, probab_threshold=0.5):
        predicted_classes = (
            LogisticRegressor.probability(x, self.w) >= probab_threshold
        ).astype(int)
        predicted_classes = predicted_classes.flatten()
        accuracy = np.mean(predicted_classes == actual_classes)
        return accuracy * 100
