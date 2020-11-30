import numpy as np


class LinearRegression(object):
    def __init__(self, iterations=1000, learning_rate=0.1, annotation=False, minimum_error=0.05, checkpoint_epoch=10):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.annotation = annotation
        self.minimum_error = minimum_error
        self.checkpoint_epoch = checkpoint_epoch
        self._theta = None
        self._cost = []


    def _add_intercept(self, X):
        theta0 = np.ones((X.shape[0]))
        return np.insert(X, 0, theta0, axis=1)


    def mse(self, X, y):
        m = X.shape[0]
        hypothesis = np.dot(X, self._theta).reshape(X.shape[0],)
        error = np.subtract(hypothesis, y)
        return np.sum(np.power(error, 2)) / m


    def fit(self, X, y):
        self._theta = np.zeros((X.shape[1], 1))
        m = X.shape[0]
        
        for epoch in range(self.iterations):
            hypothesis = np.dot(X, self._theta).reshape(X.shape[0],)
            error = np.subtract(hypothesis, y)
            gradient = np.dot(X.T, error).reshape(X.shape[1], 1)
            self._theta = self._theta - (self.learning_rate / m) * gradient

            cost = self.mse(X,y)
            self._cost.append(cost)

            if cost < self.minimum_error:
                break

            if self.annotation and epoch % self.checkpoint_epoch == 0:
                print('Cost in epoch {} = {}'.format(epoch, cost))


    def predict(self, X):
        return np.dot(X, self._theta)
