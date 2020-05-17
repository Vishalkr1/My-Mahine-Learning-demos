# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:31:44 2020

@author: Vishal
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the datasets
data = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

#cleaning the texts
import re
review = re.sub('[^a-zA-Z]',' ', data['Review'][0])
review = review.lower()


