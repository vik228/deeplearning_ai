import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import numpy as np


class XGBoostRegressor(object):
<<<<<<< HEAD
=======

>>>>>>> 6512380587113e43ce1ee996fe7953cd573a94b6
    def __init__(self, data, target, **kwargs):
        self.data = data
        self.target = target
        self._n_estimators = [None]
        self._max_depth = [None]
        self._learning_rate = [None]
        self._subsample = [None]
        self._min_child_weight = [None]
        self.xgb_regressor = None
        self.random_search_cv = None
        self._best_score = None
        self._best_params = None
        self.__dict__.update(kwargs)

    """
        The number of trees in forest
    """

    @property
    def n_estimators(self):
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, n_estimator):
        self._n_estimators = n_estimator

    """
       The maximum depth of the tree. If the max_depth is Node then nodes will
       be expanded until the leaves are pure or leaves contains less than 
       min_samples_split samples
    """

    @property
    def max_depth(self):
        return self._max_depth

    @max_depth.setter
    def max_depth(self, md):
        self._max_depth = md

    """
        Boosting learning rate 
    """

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, lr):
        self._learning_rate = lr

    """
        Subsample ratio of the training instance
    """
<<<<<<< HEAD

=======
>>>>>>> 6512380587113e43ce1ee996fe7953cd573a94b6
    @property
    def subsample(self):
        return self._subsample

    @subsample.setter
    def subsample(self, subsample):
        self._subsample = subsample

    """
        Minimum sum of instance weight(hessian) needed in a child
    """

    @property
    def min_child_weight(self):
        return self._min_child_weight

    @min_child_weight.setter
    def min_child_weight(self, min_child_weight):
        self._min_child_weight = min_child_weight

    def fit(self):
        self.xgb_regressor = xgb.XGBRegressor()
<<<<<<< HEAD
        hyperparameters = [
            a for a in dir(self) if not a.startswith("__") and not a.startswith("_")
        ]
        params = {}
        properties = [
            "data",
            "target",
            "fit",
            "predict",
            "get_metrics",
            "best_score",
            "best_params",
            "criterion",
            "xgb_regressor",
            "random_search_cv",
        ]
=======
        hyperparameters = [a for a in dir(self) if not a.startswith('__') and not a.startswith('_')]
        params = {}
        properties = ["data", "target", "fit", "predict", "get_metrics", "best_score", "best_params",
                      "criterion", "xgb_regressor", "random_search_cv"]
>>>>>>> 6512380587113e43ce1ee996fe7953cd573a94b6
        for hyperparameter in hyperparameters:
            if hyperparameter not in properties:
                params[hyperparameter] = getattr(self, hyperparameter, None)
        self.random_search_cv = RandomizedSearchCV(
            estimator=self.xgb_regressor,
            param_distributions=params,
<<<<<<< HEAD
            scoring="neg_mean_squared_error",
=======
            scoring='neg_mean_squared_error',
>>>>>>> 6512380587113e43ce1ee996fe7953cd573a94b6
            n_iter=100,
            cv=5,
            random_state=42,
            verbose=3,
<<<<<<< HEAD
            n_jobs=1,
=======
            n_jobs=1
>>>>>>> 6512380587113e43ce1ee996fe7953cd573a94b6
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
<<<<<<< HEAD
            "RMSE": np.sqrt(metrics.mean_squared_error(y_test, prediction)),
=======
            "RMSE": np.sqrt(metrics.mean_squared_error(y_test, prediction))
>>>>>>> 6512380587113e43ce1ee996fe7953cd573a94b6
        }
