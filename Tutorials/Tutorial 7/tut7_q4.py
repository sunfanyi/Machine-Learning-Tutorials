# -*- coding: utf-8 -*-
# @File    : tut7_q4.py
# @Time    : 26/11/2022
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 4: Nearest Neighbour Classification

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from tut7_tools import gen_circular_distribution, gen_sample_grid

np.random.seed(0)
X, y = gen_circular_distribution(200)
X1 = X[y == 0, :]
X2 = X[y == 1, :]

near = 3
neigh = KNeighborsClassifier(n_neighbors=near)
neigh.fit(X, y)

Xgrid, x1line, x2line = gen_sample_grid(200, 200, 10)
y_pred = neigh.predict(Xgrid).reshape(200, 200)

fig, ax = plt.subplots()
plt.contourf(x1line, x2line, y_pred)
ax.scatter(X1[:, 0], X1[:, 1])
ax.scatter(X2[:, 0], X2[:, 1])
plt.show()

