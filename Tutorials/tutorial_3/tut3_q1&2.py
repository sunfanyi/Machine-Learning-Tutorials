# -*- coding: utf-8 -*-
# @File    : tut3_q1&2.py
# @Time    : 2022/10/24
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 1: Standard linear regression
# Question 2: Higher order regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_linear_reg(xtrain, ytrain, xtest, order):
    if order == 1:
        A = [[np.sum(xtrain), np.sum(xtrain**2)],
             [len(xtrain), np.sum(xtrain)]]
        b = [np.sum(xtrain*y), np.sum(ytrain)]

        beta = np.linalg.solve(A, b)
        # Alternatively:
        # beta = np.matmul(np.linalg.inv(A), b)
        y_fitted = beta[0] + xtest * beta[1]

    elif order == 2:
        x1, x2 = xtrain, xtrain ** 2
        A = [[np.sum(x2), np.sum(x1 * x2), np.sum(x2 ** 2)],
             [np.sum(x1), np.sum(x1 ** 2), np.sum(x1 * x2)],
             [len(x1), np.sum(x1), np.sum(x2)]]
        b = [np.sum(x2 * ytrain), np.sum(x1 * ytrain), np.sum(ytrain)]

        beta = np.linalg.solve(A, b)
        y_fitted = beta[0] + xtest * beta[1] + xtest ** 2 * beta[2]

    else:
        raise Exception('order %i is not supported' % order)

    return y_fitted


if __name__ == '__main__':
    df = pd.read_csv('xray.csv')
    x = np.array(df['Distance (mm)'][:])
    y = np.array(df['Total absorption'][:])
    xtest = np.linspace(0, 6, 200)

    y_fitted_1st = fit_linear_reg(x, y, xtest, 1)
    y_fitted_2nd = fit_linear_reg(x, y, xtest, 2)

    fig, ax = plt.subplots()
    ax.scatter(x, y, s=4)
    ax.plot(xtest, y_fitted_1st, 'k')
    ax.plot(xtest, y_fitted_2nd, 'r')
    ax.text(0.5, 100, 'Fanyi Sun', size=20, zorder=0, color='#aaaaaa')
    fig.savefig('linear_reg')
    fig.show()
