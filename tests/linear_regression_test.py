import numpy as np
from utils import split_x_y
from linear_regression import LinearRegression

data = np.loadtxt('data/cateter.txt')
X,Y = split_x_y(data)
B = np.zeros(X.shape[1])

l = LinearRegression()
l.fit(X,Y,iterations=10)

# 58.0  79.0  47
print(l.predict([58.0,79.0]))