import numpy as np


class PerformanceMetrics(object):
    """
        Defines methods to implement the model

        parameters
        ----------
        y_actual : array-like, shape = [n_samples]
            Observed values from the training samples

        y_predicted : array-like, shape = [n_samples]
            Predicted value from the model
    """

    def __init__(self, y_actual, y_predicted):
        self.y_actual = y_actual
        self.y_predicted = y_predicted

    def mean_squared_error(self):
        """Compute the root mean squared error
        Returns
        ------
        rmse : root mean squared error
        """
        m = self.y_actual.shape[0]
        mse = np.sum((self.y_actual - self.y_predicted)**2)
        return np.sqrt(mse / m)

    def r2_score(self):
        """Compute the r-squared score
        Returns
        ------
        r2_score : r-squared score
        """

        ssr = np.sum((self.y_actual - self.y_predicted)**2)
        sst = np.sum((self.y_actual - np.mean(self.y_actual))**2)
        return 1 - (ssr / sst)

    def sum_of_squares_of_residuals(self):
        return np.sum((self.y_actual - self.y_predicted)**2)
