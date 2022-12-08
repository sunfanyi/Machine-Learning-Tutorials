# -*- coding: utf-8 -*-
# @File    : tut9_q1.py
# @Time    : 07/12/2022
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 1: Clustering

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN

np.random.seed(0)
n_samples = 1500
n_clusters = 6
X, y = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=1.5)

km = KMeans(n_clusters=n_clusters)

km.fit(X)
y_pred = km.predict(X)

fig, ax = plt.subplots()
fig_pred, ax_pred = plt.subplots()
for i in range(n_clusters):
    ax.scatter(X[y == i, 0], X[y == i, 1], s=8)
    ax_pred.scatter(X[y_pred == i, 0], X[y_pred == i, 1], s=8)

means = km.cluster_centers_
ax.scatter(means[:, 0], means[:, 1], color='k', s=40, marker='x')
ax_pred.scatter(means[:, 0], means[:, 1], color='k', s=40, marker='x')

fig.suptitle('True value')
fig_pred.suptitle('Predicted value')

plt.show()

