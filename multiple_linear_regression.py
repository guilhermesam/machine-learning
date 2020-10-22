import numpy as np
from utils import split_x_y


class MultipleLinearRegression(object):
    def __init__(self):
        self._intercept = None
        self._coef = None
        self.cost_history = []

    def cost_function(self, X, Y, B):
        m = len(Y)
        np.sum((X.dot(B) - Y) / (2 * m))

    def fit(self,X, Y, B, alpha, iterations):
        cost_history = [0] * iterations
        m = len(Y)
        
        for iteration in range(iterations):
            h = X.dot(B)
            loss = h - Y # Diferença entre a hipótese e o y atual
            gradient = X.T.dot(loss) / m
            B = B - alpha * gradient # Alterando valores de theta com o cálculo do gradiente
            cost = cost_function(X, Y, B) # Novo valor de custo
            cost_history[iteration] = cost
            
        return B, cost_history

    #def predict(self,X):

    def intercept(self):
        return self._intercept
    
    def coef(self):
        return self._coef


data = np.loadtxt('data/cateter.txt')
X,Y = split_x_y(data)
B = np.array([0,0,0])

l = MultipleLinearRegression()
newB, cost = l.fit(X,Y,B,0.01,10000)