# -*- coding: utf-8 -*-
# @File    : tut2_q1.py
# @Time    : 2022/10/17
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Quastion 1: Prior, posterior and likelihood calculation

import numpy as np
import matplotlib.pyplot as plt


def Gaussian(x, mu, sigma):
    g = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) / sigma) ** 2 / 2)
    return g


def calc_posterior(likelihood, prior):
    evidence = likelihood[0] * prior[0] + likelihood[1] * prior[1]
    posterior1 = (likelihood[0] * prior[0]) / evidence
    posterior2 = (likelihood[1] * prior[1]) / evidence
    return posterior1, posterior2


np.random.seed(5)
x = np.linspace(-10, 20, 200)
l1 = Gaussian(x, 2, 1.5) + Gaussian(x, 7, 0.5)
l1 /= np.trapz(l1, x)

l2 = Gaussian(x, 8, 2.5) + Gaussian(x, 3.5, 1)
l2 /= np.trapz(l2, x)

fig, ax = plt.subplots()
plt.plot(x, l1)
plt.plot(x, l2)
plt.show()


prior = [0.9, 0.1]
posterior1, posterior2 = calc_posterior([l1, l2], prior)
fig, ax = plt.subplots()
plt.plot(x, posterior1)
plt.plot(x, posterior2)
plt.xlim([-3, 15])
plt.show()
