# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:48:24 2020

@author: Vishal
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Position_Salaries.csv")

x = data.iloc[:, 1:2]
y = data.iloc[:, 2]



from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 340, random_state = 0 )
regressor.fit(x,y)

y_pred = regressor.predict([[6.5]])

# Visualising the Random forest Regression results (higher resolution)
x_grid = np.arange(min(x ['Level']), max(x['Level']), 0.01, dtype= float) 
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Random forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

