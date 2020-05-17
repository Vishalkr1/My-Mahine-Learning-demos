# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:25:37 2020

@author: Vishal
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(data.values[i,j]) for j in range(0,20)])
    
#training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#Visualising the results
results = list(rules)