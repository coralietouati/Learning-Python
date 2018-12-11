import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, SGDRegressor


# MODEL
X = 2 * np.random.rand(100, 1)
X.shape #(100, 1)
y = 4 + 3 * X + np.random.rand(100,1)
X_new =  np.array([[0],[2]])

# LINEAR REGRESSION
lin_reg = LinearRegression()
lin_reg.fit(X,y)
lin_reg.coef_, lin_reg.intercept_
lin_reg.predict(X_new)

# Stochastic Gradiant descent
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1) # 50 epochs, learning rate of 0.1
sgd_reg.fit(X,y.ravel())  # return a flatten array no []
sgd_reg.coef_, sgd_reg.intercept_

