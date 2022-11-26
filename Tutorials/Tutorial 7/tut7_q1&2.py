# -*- coding: utf-8 -*-
# @File    : tut7_q1&2.py
# @Time    : 26/11/2022
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 1: Density estimation with Parzen window
# Question 2: Window plotting

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity


def estimate_pdf_Parzen(X, x_sample, kernel, bw):

    kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(X)

    P = kde.score_samples(x_sample.reshape(len(x_sample), -1))
    # score_samples() outputs log probability, so convert using np.exp()
    p = np.exp(P)

    return p


if __name__ == '__main__':

    np.random.seed(0)
    X = np.random.normal(0, 1, [40, 1])
    x_sample = np.linspace(-5, 5, 1000)

    fig1, ax1 = plt.subplots()  # tophat
    fig2, ax2 = plt.subplots()  # gaussian

    for bw in [0.2, 1., 4.]:
        ax1.plot(x_sample, estimate_pdf_Parzen(X, x_sample, 'tophat', bw),
                 label='bandwidth %.1f' % bw)
        ax2.plot(x_sample, estimate_pdf_Parzen(X, x_sample, 'gaussian', bw),
                 label='bandwidth %.1f' % bw)

    ax1.legend()
    fig1.suptitle('tophat')
    fig1.show()

    ax2.legend()
    fig2.suptitle('gaussian')
    fig2.show()


    X = np.zeros([1, 1])
    plt.plot(x_sample, estimate_pdf_Parzen(X, x_sample, 'exponential', 0.5))
    plt.show()
