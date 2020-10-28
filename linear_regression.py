import numpy as np
from utils import split_x_y

"""
Version made without gradient descent
"""

class LinearRegression(object):
    def __init__(self):
        self._intercept = None
        self._coef = None
        self.cost_history = []

    def cost_function(self, X, Y, B):
        m = len(Y)
        return np.sum((X.dot(B) - Y) ** 2) / (2 * m)

    def fit(self, X, y):
        xT = X.T
        inversed = np.linalg.inv(xT.dot(X))
        coefficients = inversed.dot( xT ).dot(y)

        self._intercept = coefficients[0]
        self._coef = coefficients[1:]

    def predict(self, y):
        y = np.array(y)
        y_intercept = y[0]
        y_coef = y[1:]

        return y_intercept*self.intercept() + y_coef.dot(self.coef())

    def intercept(self):
        return self._intercept
    
    def coef(self):
        return self._coef


