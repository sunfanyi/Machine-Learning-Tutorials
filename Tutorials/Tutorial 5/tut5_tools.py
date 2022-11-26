# -*- coding: utf-8 -*-
# @File    : tut5_tools.py
# @Time    : 2022/11/11
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt


# define the grid to sample the domain as in previous tutorials
def gen_sample_grid(npx=200, npy=200, limit=1):
    x1line = np.linspace(-limit, limit, npx)
    x2line = np.linspace(-limit, limit, npy)
    x1grid, x2grid = np.meshgrid(x1line, x2line)
    Xgrid = np.array([x1grid, x2grid]).reshape([2, npx * npy]).T
    return Xgrid, x1line, x2line


# get a covariance matrix at an angle
def get_cov(sdx=1., sdy=1., rotangdeg=0.):
    covar = np.array([[sdx, 0], [0, sdy]])
    rot_ang = rotangdeg / 360 * 2 * np.pi
    rot_mat = np.array([[np.cos(rot_ang), -np.sin(rot_ang)],
                        [np.sin(rot_ang), np.cos(rot_ang)]])

    covar = np.matmul(np.matmul(rot_mat, covar), rot_mat.transpose())
    return covar


# generate an xor distribution
def gen_xor_distribution(n=100):
    a = np.round(n / 4).astype('int')
    b = n - a * 3
    xc1 = np.concatenate(
        [np.random.multivariate_normal([-2.3, -2.3], get_cov(0.4, 0.1, -45), a),
         np.random.multivariate_normal([2.3, 2.3], get_cov(0.4, 0.1, -45), a)])
    xc2 = np.concatenate(
        [np.random.multivariate_normal([-2.3, 2.3], get_cov(0.4, 0.1, 45), a),
         np.random.multivariate_normal([2.3, -2.3], get_cov(0.4, 0.1, 45), b)])
    xc = np.array(np.concatenate([xc1, xc2]))

    y = np.array(
        np.concatenate([np.zeros([2 * a, 1]), np.ones([a + b, 1])])).squeeze()
    X = xc
    return X, y


# generate a circular distribution of points
def gen_circular_distribution(n=500, scale=1):
    a = np.round(n / 7).astype('int')
    b = np.round(2 * n / 7).astype('int')
    c = n - a - b
    r1 = np.concatenate(
        [np.random.normal(loc=2, scale=scale, size=[a, 1]),
         np.random.normal(loc=8, scale=scale, size=[c, 1])])
    r2 = np.random.normal(loc=5, scale=scale, size=[b, 1])

    th1 = np.random.uniform(low=0, high=2 * np.pi, size=[a + c, 1])
    th2 = np.random.uniform(low=0, high=2 * np.pi, size=[b, 1])

    x1a = r1 * np.cos(th1)
    x2a = r1 * np.sin(th1)

    x1b = r2 * np.cos(th2)
    x2b = r2 * np.sin(th2)

    X = np.concatenate(
        [np.concatenate([x1a.reshape([a + c, 1]), x1b.reshape([b, 1])]),
         np.concatenate([x2a.reshape([a + c, 1]), x2b.reshape([b, 1])])],
        axis=1)

    y = np.concatenate([np.zeros([a + c, 1]), np.ones([b, 1])]).squeeze()
    return X, y
