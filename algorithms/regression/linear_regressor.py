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

    def fit(self, X, Y):
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
        m = X.shape[0]  # number of training examples
        self.cost = []
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        self.w = np.zeros((X.shape[1], 1))
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.w)
            diff = y_pred - Y
            self.w -= (np.dot(X.T, diff)) * (self.eta / m)
            cost = np.sum((diff)**2) / (2 * m)
            self.cost.append(cost)

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
