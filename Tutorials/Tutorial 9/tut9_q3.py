# -*- coding: utf-8 -*-
# @File    : tut9_q3.py
# @Time    : 07/12/2022
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 3: Mechatronics Problem

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


def gen_sample_grid(npx=200, npy=200, limit=1):
    x1line = np.linspace(limit[0], limit[1], npx)
    x2line = np.linspace(limit[2], limit[3], npy)
    x1grid, x2grid = np.meshgrid(x1line, x2line)
    Xgrid = np.array([x1grid, x2grid]).reshape([2, npx * npy]).T
    return Xgrid, x1line, x2line


df = pd.read_csv('volts.csv')
X = np.array(df.iloc[:, :2])
Y = np.array(df.iloc[:, 2:])

# Question A: Output Clustering
km = KMeans(n_clusters=3)
km.fit(Y)
class_info = km.predict(Y)

for i in range(3):
    plt.scatter(Y[class_info == i, 0], Y[class_info == i, 1], s=8)
means = km.cluster_centers_
plt.scatter(means[:, 0], means[:, 1], color='k', s=40, marker='x')
plt.title('Question A: Output Clustering')
plt.show()

print('Question A:')
print('Centroids coordinates are:')
print(means)


# Question B: Random Forest Classification
rf = ensemble.RandomForestClassifier(n_estimators=100,
                                     max_depth=20)
rf.fit(X, class_info)

fig, ax = plt.subplots()
for i in range(3):
    plt.scatter(X[class_info == i, 0], X[class_info == i, 1], s=8)
ax.set_aspect('equal')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
fig.suptitle('Question B: True Class')
fig.show()

Xgrid, x1line, x2line = gen_sample_grid(200, 200, [0, 10, 0, 10])
class_prob = rf.predict_proba(Xgrid)
class_pred = np.argmax(class_prob, axis=1)

fig2, ax2 = plt.subplots()
ax2.contourf(x1line, x2line, class_pred.reshape(200, 200))
ax2.set_aspect('equal')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
fig2.suptitle('Question B: Random Forest Classification')
fig2.show()


# Question C: Nearest Neighbours
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, class_info)

class_prob_KNN = neigh.predict_proba(Xgrid)
class_pred_KNN = np.argmax(class_prob_KNN, axis=1)

fig3, ax3 = plt.subplots()
ax3.contourf(x1line, x2line, class_pred_KNN.reshape(200, 200))
ax3.set_aspect('equal')
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
fig3.suptitle('Question C: Nearest Neighbours')
fig3.show()

