# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:35:49 2020

@author: Vishal
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("Position_Salaries.csv")
x = data.iloc[:,1:2].values
y = data.iloc[:,2].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(np.reshape(y, (10,1)))

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))

plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('truth or bluff (SVR)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
