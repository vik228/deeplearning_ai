import numpy as np
from deeplearning_ai.helpers.activation_function_helper import sigmoid


def propagate(w, b, X, Y):
    m = X.shape[1]

    #forward propagate
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)

    #compute cost
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    #backward prop
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    grads = {"dw": dw, "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["gw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print "Cost after iteration %i: %f" % (i, cost)
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    return Y_prediction
