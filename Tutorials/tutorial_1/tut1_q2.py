# -*- coding: utf-8 -*-
# @File    : tut1_q2.py
# @Time    : 2022/10/10
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 2: Make a function to generate a suitable covariance matrix

import numpy as np
import matplotlib.pyplot as plt


def get_cov(sdx=1., sdy=1., rotangdeg=0.):
    covar = np.array([[sdx ** 2, 0], [0, sdy ** 2]])
    rot_ang = rotangdeg / 360 * 2 * np.pi
    c = np.cos(rot_ang)
    s = np.sin(rot_ang)
    rot_mat = np.array([[c, -s], [s, c]])

    covar = np.matmul(np.matmul(rot_mat, covar), rot_mat.T)
    return covar


def gen_sample_grid(npx=200, npy=200, limit=1):
    x1line = np.linspace(-limit, limit, npx)
    x2line = np.linspace(-limit, limit, npy)
    x1grid, x2grid = np.meshgrid(x1line, x2line)
    Xgrid = np.array([x1grid, x2grid]).reshape([2, npx * npy]).T
    return Xgrid, x1line, x2line


if __name__ == '__main__':
    covar = get_cov(sdx=1, sdy=.3, rotangdeg=30)
    Xgrid, x1line, x2line = gen_sample_grid(npx=200, npy=200, limit=1)
    p = 1 / (2 * np.pi * np.sqrt(np.linalg.det(covar))) * np.exp(
        -1 / 2 * (np.matmul(Xgrid, np.linalg.inv(covar)) * Xgrid).sum(-1))
    p = np.reshape(p, [200, 200])
    plt.contourf(x1line, x2line, p)
    values = np.random.multivariate_normal([0, 0], covar, 100)
    plt.scatter(values[:, 0], values[:, 1])
    plt.axis('square')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()
