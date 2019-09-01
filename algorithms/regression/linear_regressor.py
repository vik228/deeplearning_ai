import numpy as np


class LinearRegressor(object):
    """ Linear Regression using Gradient Descent.
    Parameters
    ---------
    eta: float
        Learning Rate
    n_iterations: int
        Number of passes over training set

    Attributes
    ---------
    w_ : weights/ after fitting the model
    cost: total error of the model after each iterations
    """

    def __init__(self, eta=0.05, n_iterations=1000):
        self.eta = eta
        self.n_iterations = n_iterations
        self.cost = None
        self.w = None

    def fit(self, x, y):
        """Fit The training data
        Parameters
        ----------
        X: numpy-array, shape = [n_samples, n_features]
            Training samples
        Y: numpy-array, shape = [n_samples, 1]
            Target values

        Returns
        -------
        self: object
        """
        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.eta / m) * gradient_vector
            cost = np.sum((residuals**2)) / (2 * m)
            self.cost_.append(cost)
        return self

    def predict(self, X):
        """Pridict the value after the model is trained
        Parameters
        ----------
        X: numpy-array, shape = [n_samples, n_features]
            Training samples

        Returns
        -------
        Predicted Values
        """
        return np.dot(X, self.w)
