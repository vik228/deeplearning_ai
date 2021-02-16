from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from utils.utils import timer
from sklearn import metrics
import numpy as np
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
import pydotplus


class MyDecisionTreeRegressor(object):

    def __init__(self, data, target, **kwargs):
        self.data = data
        self.target = target
        self._criterion = "mse"
        self._splitter = ["best", "random"]
        self._max_depth = [3, 4, 5, 6, 8, 10, 12, 15]
        self._min_samples_split = [2, 3, 4, 5]
        self._min_samples_leaf = [1, 2, 3, 4, 5]
        self._max_leaf_nodes = [None, 10, 20, 30, 40, 50, 60, 70]
        self._max_features = ["auto", "log2", "sqrt", None]
        self._best_params = {}
        self._best_score = None
        self.grid_search = None
        self.dtree = None
        self.__dict__.update(kwargs)

    """
        The function used to measure the quality of the split. Supported methods are
        mse -> Mean Squared Error
        friedman_mse -> This uses Mean Squared Error with friedman's improvement score for potential
        splits
        mae -> Mean Absolute Error
    """

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, c):
        self._criterion = c

    """
        The strategies used to choose the split. There are two methods best and random. 
        Best to choose the best split and random to choose the random best split 
    """

    @property
    def splitter(self):
        return self._splitter

    @splitter.setter
    def splitter(self, splitter):
        self._splitter = splitter

    """
        The maximum depth of the tree. If the max_depth is Node then nodes will
        be expanded until the leaves are pure or leaves contains less than min_samples_split 
        samples
    """

    @property
    def max_depth(self):
        return self._max_depth

    @max_depth.setter
    def max_depth(self, max_depth):
        self._max_depth = max_depth

    """
       The minimum number of samples required to split an internal node
    """

    @property
    def min_samples_split(self):
        return self._min_samples_split

    @min_samples_split.setter
    def min_samples_split(self, min_samples_split):
        self._min_samples_split = min_samples_split

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

    @property
    def best_params(self):
        return self._best_params

    @property
    def best_score(self):
        return self._best_score

    def fit(self):
        self.dtree = DecisionTreeRegressor(self.criterion)
        hyperparameters = [a for a in dir(self) if not a.startswith('__') and not a.startswith('_')]
        params = {}
        properties = ["data", "target", "fit", "predict", "get_metrics", "plot_tree", "best_score", "best_params", "criterion", "dtree", "grid_search"]
        for hyperparameter in hyperparameters:
            if hyperparameter not in properties:
                params[hyperparameter] = getattr(self, hyperparameter, None)
        self.grid_search = GridSearchCV(
            self.dtree,
            param_grid=params,
            scoring="neg_mean_squared_error",
            n_jobs=1,
            cv=10,
            verbose=3,
        )
        start_time = timer(None)
        self.grid_search.fit(self.data, self.target)
        timer(start_time)
        self._best_params = self.grid_search.best_params_
        self._best_score = self.grid_search.best_score_

    def predict(self, X_test):
        return self.grid_search.predict(X_test)

    @staticmethod
    def get_metrics(y_test, prediction):
        return {
            "MAE": metrics.mean_absolute_error(y_test, prediction),
            "MSE": metrics.mean_squared_error(y_test, prediction),
            "RMSE": np.sqrt(metrics.mean_squared_error(y_test, prediction))
        }

    def plot_tree(self):
        features = list(self.data.columns)
        dot_data = StringIO()
        export_graphviz(
            self.dtree, out_file=dot_data, feature_names=features, filled=True, rounded=True
        )
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        Image(graph.create_png())






