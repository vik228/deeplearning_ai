from algorithms.base import KNeighbors

<<<<<<< HEAD

class KNeighborsRegressor(KNeighbors):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X_train, y_train):
        super().fit(X_train, y_train)

    def func(self, df_nn):
        return df_nn["dist"].mean()

    def predict(self, x_test):
        return super().predict(x_test, self.func)
=======
class KNeighborsRegressor(KNeighbors):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def fit(self, X_train, y_train):
        super().fit(X_train, y_train)
    
    def func(self, df_nn):
        return df_nn["dist"].mean()
    
    def predict(self, x_test):
        return super().predict(x_test, self.func)
>>>>>>> 6512380587113e43ce1ee996fe7953cd573a94b6
