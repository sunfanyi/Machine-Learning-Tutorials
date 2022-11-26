# -*- coding: utf-8 -*-
# @File    : tut4_q2.py
# @Time    : 2022/11/4
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 2: Plotting linear discriminant functions

import numpy as np
import matplotlib.pyplot as plt


def linear_discriminant(x1, x2):
    x1grid, x2grid = np.meshgrid(x1, x2)
    Xgrid = np.array([x1grid, x2grid]).reshape(2, n**2).T

    omega = np.array([-1, -3])
    omega0 = 1

    g = np.matmul(Xgrid, omega) + omega0
    g = np.reshape(g, [n, n])

    fig, ax = plt.subplots()
    ax.contourf(x1grid, x2grid, g)
    fig.colorbar(None)
    fig.show()
    return fig, ax


if __name__ == '__main__':
    n = 20
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
    fig, ax = linear_discriminant(x1, x2)
    x1line = x1
    x2line = -1/3 * x1 + 1/3
    ax.plot(x1line, x2line, 'r')
    fig.show()
