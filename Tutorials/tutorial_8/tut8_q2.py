# -*- coding: utf-8 -*-
# @File    : tut8_q2.py
# @Time    : 01/12/2022
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 2: Random Forest Classifier

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import ensemble

sys.path.append('..')
from tools import gen_sample_grid


def visualise_random_forest(m, num_trees, max_depth):
    np.random.seed(0)
    X, y = datasets.make_classification(n_samples=m, n_features=2,
                                        n_informative=2, n_redundant=0)

    rf = ensemble.RandomForestClassifier(n_estimators=num_trees,
                                         max_depth=max_depth)
    rf.fit(X, y)

    Xgrid, x1line, x2line = gen_sample_grid(200, 200, 4)
    prob = rf.predict_proba(Xgrid)[:, 0]

    fig, ax = plt.subplots()
    contourf_ = ax.contourf(x1line, x2line, prob.reshape(200, 200))
    fig.colorbar(contourf_)
    ax.scatter(X[y == 0, 0], X[y == 0, 1])
    ax.scatter(X[y == 1, 0], X[y == 1, 1])
    fig.show()


if __name__ == '__main__':
    m = 200
    num_trees = 100
    max_depth = 20
    visualise_random_forest(m, num_trees, max_depth)
