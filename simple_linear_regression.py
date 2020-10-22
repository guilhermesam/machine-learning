import numpy as np

class SimpleLinearRegression(object):

    def __init__(self):
        self._coef = None
        self._intercept = None

    def hypothesis(self,theta0, theta1, x):
        return theta0 + theta1 * x

    def fit(self, X, y):
        n = len(x)
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)

        numerator = 0
        denominator = 0
        for i in range(n):
            numerator += (X[i] - x_mean) * (y[i] - y_mean)
            denominator += (X[i] - x_mean) ** 2

        self._coef = numerator / denominator
        self._intercept = y_mean - (self._coef * x_mean)

    def coef(self):
        return self._coef
    
    def intercept(self):
        return self._intercept

    def predict(self, X):
        predictions = []
        for i in X:
            predictions.append(self.hypothesis(self._intercept, self._coef, i))

        return predictions