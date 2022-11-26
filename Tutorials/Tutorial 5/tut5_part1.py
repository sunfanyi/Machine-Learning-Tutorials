# -*- coding: utf-8 -*-
# @File    : tut5_part1.py
# @Time    : 2022/11/11
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Part 1: Support Vector Classification using sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from tut5_tools import gen_xor_distribution, gen_sample_grid

np.random.seed(0)
X, y = gen_xor_distribution(400)

fig, ax = plt.subplots()
# ax.scatter(X[y == 0, 0], X[y == 0, 1])
# ax.scatter(X[y == 1, 0], X[y == 1, 1])
# fig.show()

svm = SVC(C=1000, gamma='auto', kernel='poly', degree=2)
svm.fit(X, y)

npx, npy = 200, 200
Xgrid, x1line, x2line = gen_sample_grid(npx, npy, limit=4)
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
