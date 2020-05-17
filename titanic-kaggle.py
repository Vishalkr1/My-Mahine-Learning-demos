# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:47:09 2020

@author: Vishal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Handle features of training data
train_data = pd.read_csv("train.csv")
missing_values = train_data.isnull().sum()
sns.heatmap(train_data.isnull(), yticklabels = False, cbar = False)

train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())
train_data["Embarked"] = train_data["Embarked"].fillna(train_data["Embarked"].mode()[0])

train_data.drop(["Name", "Cabin", "PassengerId", "Ticket"], axis = 1, inplace = True)

x_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
laberencoder_x = LabelEncoder()
x_train[:,1] = laberencoder_x.fit_transform(x_train[:, 1])
x_train[:,6] = laberencoder_x.fit_transform(x_train[:, 6])

onehotencoder = OneHotEncoder(categorical_features = [0])
onehotencoder = OneHotEncoder(categorical_features = [6])
x_train1 = onehotencoder.fit_transform(x_train).toarray()

import statsmodels.formula.api as sm
x_opt = x_train1[:, [0, 1, 2, 3, 4, 5, 6]]
OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
OLS.summary()

#Handle features of test data
test_data = pd.read_csv("test.csv")
missing_values_test_data = test_data.isnull().sum()
passengerID = test_data.iloc[:,0].values
passengerID = pd.DataFrame(passengerID)
test_data.drop(["Name", "Cabin", "PassengerId", "Ticket"], axis = 1, inplace = True)
test_data["Age"] = test_data["Age"].fillna(test_data["Age"].mean())
test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].mean())

test_data = test_data.iloc[:,:].values
laberencoder_test = LabelEncoder()
test_data[:,1] = laberencoder_test.fit_transform(test_data[:, 1])
test_data[:,6] = laberencoder_test.fit_transform(test_data[:, 6])

onehotencoder = OneHotEncoder(categorical_features = [0])
onehotencoder = OneHotEncoder(categorical_features = [1])
onehotencoder = OneHotEncoder(categorical_features = [6])
test1 = onehotencoder.fit_transform(test_data).toarray()

test1 = test1[:, [0, 1, 2, 3, 4, 5, 6]]

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_opt, y_train)

y_pred = classifier.predict(test1)
pred = pd.DataFrame(y_pred)
dataset = pd.concat([passengerID, pred], axis = 1)
dataset.columns = ["PassengerId", "Survived"]
dataset.to_csv("submission1.csv", index = False)
#68.889% accuracy

from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier1.fit(x_opt, y_train)

y_pred1 = classifier1.predict(test1)

y_pred1 = classifier.predict(test1)
pred = pd.DataFrame(y_pred1)
dataset = pd.concat([passengerID, pred], axis = 1)
dataset.columns = ["PassengerId", "Survived"]
dataset.to_csv("submission2.csv", index = False)
#68.889% accuracy

from sklearn.naive_bayes import GaussianNB
classifier2 = GaussianNB()
classifier2.fit(x_opt, y_train)

# Predicting the Test set results
y_pred2 = classifier2.predict(test1)
pred = pd.DataFrame(y_pred1)
dataset = pd.concat([passengerID, pred], axis = 1)
dataset.columns = ["PassengerId", "Survived"]
dataset.to_csv("submission3.csv", index = False)