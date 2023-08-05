import numpy as np
import pandas as pd
from collections import Counter

<<<<<<< HEAD

class KNeighbors(object):
    def __init__(self, **kwargs):
        self.n_neighbors = kwargs.get("n_neighbors")
        self.p = kwargs.get("p")
        self.X_train = None
        self.y_train = None

    @staticmethod
    def minkowski_distance(a, b, p=1):
        return np.sum(np.abs(a - b) ** p) ** (1 / p)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

=======
class KNeighbors(object):

    def __init__(self, **kwargs):
        self.n_neighbors = kwargs.get('n_neighbors')
        self.p = kwargs.get('p')
        self.X_train = None
        self.y_train = None
    
    @staticmethod
    def minkowski_distance(a, b, p = 1):
        return np.sum(np.abs(a - b)**p)**(1/p)
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
>>>>>>> 6512380587113e43ce1ee996fe7953cd573a94b6
    def predict(self, x_test, func):
        preds = []
        for test_point in x_test:
            distances = []
            for train_point in self.X_train:
<<<<<<< HEAD
                distances.append(
                    KNeighbors.minkowski_distance(test_point, train_point, p=self.p)
                )
            df_dists = pd.DataFrame(
                distances, index=self.y_train.index, columns=["dist"]
            )
            df_nn = df_dists.sort_values(by=["dist"], inplace=True)[: self.n_neighbors]
=======
                distances.append(KNeighbors.minkowski_distance(test_point, train_point, p = self.p))
            df_dists = pd.DataFrame(distances, index=self.y_train.index, columns=['dist'])
            df_nn = df_dists.sort_values(by=['dist'], inplace=True)[:self.n_neighbors]
>>>>>>> 6512380587113e43ce1ee996fe7953cd573a94b6
            preds.append(func(df_nn))
        return preds
