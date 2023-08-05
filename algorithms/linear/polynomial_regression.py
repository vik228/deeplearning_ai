from algorithms.regression.linear_regressor import LinearRegressor
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegressor(object):
    def __init__(self, eta=0.05, n_iterations=1000, degree=1):
        self.eta = eta
        self.n_iterations = n_iterations
        self.degree = degree
        self.linear_regressor = LinearRegressor()

    def polynomial_features(self, x):
        polynomial_features = PolynomialFeatures(degree=self.degree)
        return polynomial_features.fit_transform(x)

    def fit(self, x, y):
        x_poly = self.polynomial_features(x)
        self.linear_regressor.fit(x_poly, y)

    def predict(self, x):
        x_poly = self.polynomial_features(x)
        return self.linear_regressor.predict(x_poly)
