# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 09:19:31 2020

@author: Vishal
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Mall_Customers.csv")
x = data.iloc[:, [3, 4]].values

# choosing optimal no of custers :: Elbow method

from sklearn.cluster import KMeans
wcss = []
for i in range( 1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("The Elbow method")
plt.xlabel("no of clusters")
plt.ylabel("wcss")
plt.show()

# Applying k-means to dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

# Visualising Clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = "red", label = "cluster1")
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = "green", label = "cluster2")
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = "blue", label = "cluster3")
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = "orange", label = "cluster4")
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = "pink", label = "cluster5")

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1] , s = 300, c = "yellow", label = "centroids")
plt.title("Cluster of clients")
plt.xlabel("Annual income")
plt.ylabel("spending score")
plt.legend()
plt.show()