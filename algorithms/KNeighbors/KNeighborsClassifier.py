from algorithms.base import KNeighbors
from collections import Counter

<<<<<<< HEAD

class KNeighborsClassifier(KNeighbors):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fit(self, X_train, y_train):
        super().fit(X_train, y_train)

    def func(self, df_nn):
        counter = Counter(self.y_train[df_nn.index])
        return counter.most_common()[0][0]

    def predict(self, x_test):
        return super().predict(x_test, self.func)
=======
class KNeighborsClassifier(KNeighbors):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    

    def fit(self, X_train, y_train):
        super().fit(X_train, y_train)
    
    def func(self, df_nn):
        counter = Counter(self.y_train[df_nn.index])
        return counter.most_common()[0][0]
    
    def predict(self, x_test):
        return super().predict(x_test, self.func)







>>>>>>> 6512380587113e43ce1ee996fe7953cd573a94b6
