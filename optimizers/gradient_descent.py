import numpy as np


class GradientDescent(object):

    def __init__(self, learning_rate, X, Y, **kwargs):
        self.theta_history = None
        self.cost_history = None
        self.learning_rate = learning_rate
        self.X = X
        self.Y = Y
        self._iterations = None
        self._theta = None
        self._callback = None
        self.__dict__.update(kwargs)

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, callback):
        self._callback = callback

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, iterations):
        self._iterations = iterations

    def cost(self):
        m = len(self.Y)
        predictions = np.dot(self.X, self._theta)
        cost = (1 / 2 * m) * np.sum(np.square(predictions - self.Y))
        return cost

    def cal_cost(self, X, Y):
        m = len(Y)
        predictions = np.dot(X, self._theta)
        cost = (1 / 2 * m) * np.sum(np.square(predictions - Y))
        return cost

    def optimise(self):
        self.theta_history = np.zeros((self.iterations, self.X.shape[1]))
        self.cost_history = np.zeros(self.iterations)
        m = len(self.Y)
        for it in range(self.iterations):
            predictions = np.dot(self.X, self.theta)
            self.theta = self.theta - (1 / m) * self.learning_rate * (np.dot(self.X.T, predictions - self.Y))
            self.theta_history[it, :] = self.theta.T
            self.cost_history[it] = self.cost()
            if getattr(self, 'callback', None):
                self.callback(predictions, self.cost_history)

    def optimise_sgd(self):
        self.cost_history = np.zeros(self.iterations)
        m = len(self.Y)
        for it in range(self.iterations):
            cost = 0.0
            for i in range(m):
                rand_ind = np.random.randint(0, m)
                x_i = self.X[rand_ind, :].reshape(1, self.X.shape[1])
                y_i = self.Y[rand_ind].reshape(1, 1)
                predictions = np.dot(x_i, self.theta)
                self.theta = self.theta - (1 / m) * self.learning_rate * (np.dot(x_i.T, predictions - y_i))
                cost += self.cal_cost(x_i, y_i)
            self.cost_history[it] = cost

    def optimise_mini_batch_gd(self, **kwargs):
        batch_size = kwargs.get('batch_size', 20)
        cost_func = kwargs.get('func', None)
        gradient = kwargs.get('gradient', None)
        self.cost_history = np.zeros(self.iterations)
        m = len(self.Y)
        for it in range(self.iterations):
            cost = 0.0
            indices = np.random.permutation(m)
            X = self.X[indices]
            Y = self.Y[indices]
            for i in range(0, m, batch_size):
                x_i = X[i:i + batch_size]
                y_i = Y[i:i + batch_size]
                predictions = np.dot(x_i, self.theta)
                if gradient:
                    self.theta = self.theta - self.learning_rate*gradient(self.theta, x_i, y_i)
                else:
                    self.theta = self.theta - (1 / m) * self.learning_rate * (np.dot(x_i.T, predictions - y_i))
                if cost_func:
                    cost += cost_func(self.theta, x_i, y_i)
                else:
                    cost += self.cal_cost(x_i, y_i)
            self.cost_history[it] = cost

    def predict(self):
        return np.dot(self.X, self.theta)
