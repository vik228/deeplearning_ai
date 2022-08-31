from algorithms.base import KNeighbors

class KNeighborsRegressor(KNeighbors):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def fit(self, X_train, y_train):
        super().fit(X_train, y_train)
    
    def func(self, df_nn):
        return df_nn["dist"].mean()
    
    def predict(self, x_test):
        return super().predict(x_test, self.func)