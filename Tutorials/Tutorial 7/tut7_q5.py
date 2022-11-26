# -*- coding: utf-8 -*-
# @File    : tut7_q5.py
# @Time    : 26/11/2022
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 5: Estimate Density Manually

import numpy as np
import matplotlib.pyplot as plt

from tut7_tools import gen_circular_distribution, gen_sample_grid

np.random.seed(0)
X, y = gen_circular_distribution(2000, scale=1, scale_ellipse=1.2)
X = X[y == 1, :]

n_gird = 20
limit = 10
h = limit * 2 / n_gird
Xgrid, x1line, x2line = gen_sample_grid(n_gird+1, n_gird+1, limit)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1])
ticks_top = np.linspace(-limit, limit, n_gird+1)
ax.set_xticks(ticks_top)
ax.set_yticks(ticks_top)
ax.grid()
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_aspect('equal')
plt.show()

# convert values to the index of the grids
# eg., in linspace(-10, 10, 20), x = 6.5 belongs to the 16th grid
x_idx = (np.round((X[:, 0] - (-limit)) / h)).astype('int')
y_idx = (np.round((X[:, 1] - (-limit)) / h)).astype('int')

density = np.zeros([n_gird+1, n_gird+1])

for i in range(len(X)):
    density[x_idx[i], y_idx[i]] += 1

totDen = np.sum(density[:])
density /= totDen * h * h  # 2 dimension so h**2

fig, ax = plt.subplots()
contourf_ = ax.contourf(x1line, x2line, density.T)
fig.colorbar(contourf_)
# ax.plot(X[:, 0], X[:, 1], 'rx')
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_aspect('equal')
fig.show()
