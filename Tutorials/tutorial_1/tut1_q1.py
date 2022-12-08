# -*- coding: utf-8 -*-
# @File    : tut1_q1.py
# @Time    : 2022/10/10
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 1: Generate and plot a test dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(0)
X, y = datasets.make_classification(n_samples=100,
                                    n_features=2,
                                    n_informative=2,
                                    n_redundant=0)
X[:, 0] = np.abs(X[:, 0] * 0.5 + 5)
X[:, 1] = np.abs(X[:, 1] * 30 + 160)

fig, ax = plt.subplots()
ax.scatter(X[y == 0, 0], X[y == 0, 1])
ax.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
x1 = np.linspace(3, 7.5, 50)
x2 = -280 * x1 + 1400
plt.plot(x1, x2)
plt.show()


