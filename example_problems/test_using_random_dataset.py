import numpy as np
from algorithms.regression.linear_regressor import LinearRegressor
from algorithms.regression.polynomial_regression import PolynomialRegressor
from algorithms.matrices import PerformanceMetrics


def test_using_random_dataset():
    np.random.seed(0)
    x = 2 - 3 * np.random.normal(0, 1, 20)
    y = x - 2 * (x**2) + 0.5 * (x**3) + np.random.normal(-3, 3, 20)
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    model = LinearRegressor()
    model.fit(x, y)
    y_pred = model.predict(x)
    model_poly = PolynomialRegressor(degree=2)
    model_poly.fit(x, y)
    y_pred_poly = model_poly.predict(x)
