# -*- coding: utf-8 -*-
# @File    : tut5_part2.py
# @Time    : 2022/11/12
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Part 2: SVC on a circular distribution

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from tut5_tools import gen_circular_distribution, \
    gen_xor_distribution, gen_sample_grid

np.random.seed(0)
X, y = gen_circular_distribution(200)

fig, ax = plt.subplots()
# ax.scatter(X[y == 0, 0], X[y == 0, 1])
# ax.scatter(X[y == 1, 0], X[y == 1, 1])
# fig.show()

svm = SVC(C=1, gamma='auto', kernel='rbf')
svm.fit(X, y)

npx, npy = 200, 200
Xgrid, x1line, x2line = gen_sample_grid(npx, npy, limit=10)
Z = np.reshape(svm.decision_function(Xgrid), [npx, npy])
ax.contour(x1line, x2line, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])  # [-margin, decision boundary, margin]

class_vals = svm.predict(Xgrid)
class_vals = np.reshape(class_vals, [npx, npy])

ax.contourf(x1line, x2line, class_vals)
ax.scatter(X[y == 0, 0], X[y == 0, 1])
ax.scatter(X[y == 1, 0], X[y == 1, 1])
fig.show()

sv = svm.support_vectors_

ax.scatter(sv[:, 0], sv[:, 1], marker="x", c="#000000")
fig.show()



