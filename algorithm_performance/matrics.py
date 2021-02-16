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
        rmse : mean squared error
        """
        m = self.y_actual.shape[0]
        mse = self.sum_of_squares_of_residuals()
        return mse / m

    def mean_absolute_error(self):
        """Compute the mean absolute error
            Returns
            ------
            mae : mean_absolute_error
        """
        n = self.y_actual.shape[0]
        mae = np.sum(np.abs(self.y_predicted - self.y_actual))
        return mae / n

    def root_mean_squared_error(self):
        """Compute the root mean squared error
            Returns
            ------
            rmse : root mean squared error
        """
        return np.sqrt(self.mean_squared_error())

    def r2_score(self):
        """Compute the r-squared score
        Returns
        ------
        r2_score : r-squared score

        RSS -> Sum of squares of residuals
        TSS -> Total sum of squares
        """

        RSS = self.sum_of_squares_of_residuals()
        TSS = self.total_sum_of_squares()
        return 1 - (RSS / TSS)

    def total_sum_of_squares(self):
        return np.sum((self.y_actual - np.mean(self.y_actual)) ** 2)

    def sum_of_squares_of_residuals(self):
        return np.sum((self.y_actual - self.y_predicted) ** 2)
