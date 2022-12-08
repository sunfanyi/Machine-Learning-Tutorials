# -*- coding: utf-8 -*-
# @File    : tut7_q3.py
# @Time    : 26/11/2022
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 3: 2D Function Classification

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

sys.path.append('..')
from tools import gen_circular_distribution, gen_sample_grid


def kde_classification(bw, X1, X2, X_sample):
    kde1 = KernelDensity(kernel='gaussian', bandwidth=bw)
    kde2 = KernelDensity(kernel='gaussian', bandwidth=bw)

    kde1.fit(X1)
    kde2.fit(X2)

    p1 = kde1.score_samples(X_sample)
    p1 = np.exp(p1)
    p2 = kde2.score_samples(X_sample)
    p2 = np.exp(p2)

    class_info = p1 > p2

    return class_info


if __name__ == '__main__':
    np.random.seed(0)
    X, y = gen_circular_distribution(200)
    X1 = X[y == 0, :]
    X2 = X[y == 1, :]
    Xgrid, x1line, x2line = gen_sample_grid(200, 200, 10)

    bw = 1
    class_info = kde_classification(bw, X1, X2, Xgrid)
    class_info = class_info.reshape(200, 200)
    fig, ax = plt.subplots()
    ax.contourf(x1line, x2line, class_info)
    ax.scatter(X1[:, 0], X1[:, 1])
    ax.scatter(X2[:, 0], X2[:, 1])
    fig.suptitle('bandwidth = %.1f' % bw)
    fig.show()

    bw = 0.5
    class_info = kde_classification(bw, X1, X2, Xgrid)
    class_info = class_info.reshape(200, 200)
    fig, ax = plt.subplots()
    ax.contourf(x1line, x2line, class_info)
    ax.scatter(X1[:, 0], X1[:, 1])
    ax.scatter(X2[:, 0], X2[:, 1])
    fig.suptitle('bandwidth = %.1f' % bw)
    fig.show()
