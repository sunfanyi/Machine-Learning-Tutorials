# -*- coding: utf-8 -*-
# @File    : tut8_q1.py
# @Time    : 01/12/2022
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 1: Decision Tree Classifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree

from tut8_tools import gen_sample_grid


if __name__ == '__main__':
    np.random.seed(0)
    X, y = datasets.make_classification(n_samples=100, n_features=2,
                                        n_informative=2, n_redundant=0)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)

    Xgrid, x1line, x2line = gen_sample_grid(200, 200, 4)
    class_vals = clf.predict(Xgrid).reshape(200, 200)

    fig, ax = plt.subplots()
    ax.contourf(x1line, x2line, class_vals)
    ax.scatter(X[y == 0, 0], X[y == 0, 1])
    ax.scatter(X[y == 1, 0], X[y == 1, 1])
    ax.set_aspect('equal')
    fig.show()
