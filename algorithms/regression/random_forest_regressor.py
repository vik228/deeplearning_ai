from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn import metrics


class MyRandomForestRegressor(object):

    def __init__(self, data, target, **kwargs):
        self.data = data
        self.target = target
        self._n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
        self._criterion = "mse"
        self._max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
        self._min_samples_split = [2, 5, 10, 15, 100]
        self._min_samples_leaf = [1, 2, 5, 10]
        self._max_leaf_nodes = [None, 10, 20, 30, 40, 50, 60, 70]
        self._max_features = ["auto", "sqrt"]
        self._best_params = {}
        self._best_score = None
        self.random_forest = None
        self.random_search_cv = None
        self.__dict__.update(kwargs)

    """
        The number of trees in forest
    """

    @property
    def n_estimators(self):
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, e):
        self._n_estimators = e

    """
       The function used to measure the quality of the split. Supported methods are
       mse -> Mean Squared Error
       mae -> Mean Absolute Error
   """

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, c):
        self._criterion = c

    """
       The maximum depth of the tree. If the max_depth is Node then nodes will
       be expanded until the leaves are pure or leaves contains less than min_samples_split 
       samples
    """

    @property
    def max_depth(self):
        return self._max_depth

    @max_depth.setter
    def max_depth(self, m_depth):
        self._max_depth = m_depth

    """
        The minimum number of samples required to split an internal node
    """

    @property
    def min_samples_split(self):
        return self._min_samples_split

    @min_samples_split.setter
    def min_samples_split(self, m_samples_split):
        self._min_samples_split = m_samples_split

    """
        The minimum number of samples required to be at a leaf node.
    """

    @property
    def min_samples_leaf(self):
        return self._min_samples_leaf

    @min_samples_leaf.setter
    def min_samples_leaf(self, min_samples_leaf):
        self._min_samples_leaf = min_samples_leaf

    """
        The number of features to consider when looking for the best split. The values can be
        int, float or {"auto", "sqrt", "log2"}
    """

    @property
    def max_features(self):
        return self._max_features

    @max_features.setter
    def max_features(self, max_features):
        self._max_features = max_features

    """
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes
    """

    @property
    def max_leaf_nodes(self):
        return self._max_leaf_nodes

    @max_leaf_nodes.setter
    def max_leaf_nodes(self, max_leaf_nodes):
        self._max_leaf_nodes = max_leaf_nodes


    def fit(self):
        self.random_forest = RandomForestRegressor(self.criterion)
        hyperparameters = [a for a in dir(self) if not a.startswith('__') and not a.startswith('_')]
        params = {}
        properties = ["data", "target", "fit", "predict", "get_metrics", "best_score", "best_params",
                      "criterion", "random_forest", "random_search_cv"]
        for hyperparameter in hyperparameters:
            if hyperparameter not in properties:
                params[hyperparameter] = getattr(self, hyperparameter, None)
        self.random_search_cv = RandomizedSearchCV(
            estimator=self.random_forest,
            param_distributions=params,
            scoring='neg_mean_squared_error',
            n_iter=100,
            cv=5,
            random_state=42,
            verbose=3,
            n_jobs=1
        )
        self.random_search_cv.fit(self.data, self.target)
        self._best_score = self.random_search_cv.best_score_
        self._best_params = self.random_search_cv.best_params_

    def predict(self, X_test):
        return self.random_search_cv.predict(X_test)

    @staticmethod
    def get_metrics(y_test, prediction):
        return {
            "MAE": metrics.mean_absolute_error(y_test, prediction),
            "MSE": metrics.mean_squared_error(y_test, prediction),
            "RMSE": np.sqrt(metrics.mean_squared_error(y_test, prediction))
        }
