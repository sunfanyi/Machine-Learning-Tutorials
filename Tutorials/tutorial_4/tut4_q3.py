# -*- coding: utf-8 -*-
# @File    : tut4_q3.py
# @Time    : 2022/11/4
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 3: Plotting classification areas

import sys
import numpy as np
import matplotlib.pyplot as plt


def multi_category_classification_2D(weighting_terms, npx, npy, Xgrid, Ygrid):
    linear_discriminant_functions = []
    for a in weighting_terms:
        a = np.array(a).reshape(-1, 1)

        if a.shape != (Ygrid.shape[1], 1):
            print("Error!! Shape of the weighting term is incorrect.")
            sys.exit()
        if Xgrid.shape != (npx * npy, 2):
            print("Error!! Shape of Xgrid is incorrect.")
            sys.exit()

        # calculate each of the functions
        g = np.matmul(Ygrid, a)
        linear_discriminant_functions.append(g)

    # combine all functions together
    gconc = np.concatenate(linear_discriminant_functions, axis=1)

    # find which of the values is largest for each row (return the index)
    omega = np.argmax(gconc, axis=1)  # omega represents the class

    # put back onto 2D grid so it can easily be plotted
    omega = np.reshape(omega, [npx, npy])

    return omega



if __name__ == '__main__':
    npx = 200
    npy = 200
    x1 = np.linspace(0, 1, npx)
    x2 = np.linspace(0, 1, npy)
    x1grid, x2grid = np.meshgrid(x1, x2)
    Xgrid = np.array([x1grid, x2grid]).reshape(2, npx * npy).T

    # ================================ Case 1 ================================
    weighting_terms = [[1.3, -1, -3],
                       [-2, 1, 2],
                       [0.3, 0.1, -0.1],
                       [0, -1, 1],
                       [-0.2, 1.5, -1]]
    # Ygrid is defined as the same as Xgrid, except it has 1 at the beginning
    Ygrid = np.concatenate([np.ones([npx * npy, 1]), Xgrid], axis=1)

    omega = multi_category_classification_2D(weighting_terms, npx, npy,
                                             Xgrid, Ygrid)
    fig, ax = plt.subplots()
    contourf_ = ax.contourf(x1grid, x2grid, omega)
    fig.colorbar(contourf_)
    fig.suptitle('y = (1, x1, x2)\'; 5 classes')
    fig.show()

    # ================================ Case 2 ================================
    weighting_terms = [[1, -1, -3],
                       [-1, 1, 3]]
    Ygrid = np.concatenate([np.ones([npx * npy, 1]), Xgrid], axis=1)

    omega = multi_category_classification_2D(weighting_terms, npx, npy,
                                             Xgrid, Ygrid)
    fig, ax = plt.subplots()
    contourf_ = ax.contourf(x1grid, x2grid, omega)
    fig.colorbar(contourf_)
    fig.suptitle('y = (1, -1*x1, -3*x2)\'; 2 classes')
    fig.show()

    # ================================ Case 3 ================================
    weighting_terms = [[1.3, -1, -3, -10],
                       [-1, 1.5, 3, -1],
                       [0.4, -0.1, -0.1, 3],
                       [0.5, -1, 1, -0.1],
                       [-0.2, 1.5, -1, 0.4]]
    Ygrid = np.concatenate([np.ones([npx * npy, 1]), Xgrid,
                            (Xgrid[:, 0] * Xgrid[:, 1]).reshape(-1, 1)], axis=1)
    # Ygrid = np.concatenate([np.ones([npx * npy, 1]), Xgrid,
    #                         np.array([Xgrid[:, 0] * Xgrid[:, 1]]).T], axis=1)

    omega = multi_category_classification_2D(weighting_terms, npx, npy,
                                             Xgrid, Ygrid)
    fig, ax = plt.subplots()
    contourf_ = ax.contourf(x1grid, x2grid, omega)
    fig.colorbar(contourf_)
    fig.suptitle('y = (1, x1, x2, x1x2)\'; 5 classes')
    fig.show()
