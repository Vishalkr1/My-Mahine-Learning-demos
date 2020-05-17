# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:18:33 2020

@author: Vishal
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

""" Part-1 Data Preprocessing"""

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values

missing_values = pd.DataFrame(X).isnull().sum()

#handling categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
X[:,1] = labelencoder_x1.fit_transform(X[:, 1])
labelencoder_x2 = LabelEncoder()
X[:,2] = labelencoder_x1.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
"""Part-2 Making ANN"""
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising ANN
classifier = Sequential()

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
for i in range(0,2000):
    if(y_pred[i,0] > 0.5):
        y_pred[i,0] = 1
    else:
        y_pred[i,0] = 0
        

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)