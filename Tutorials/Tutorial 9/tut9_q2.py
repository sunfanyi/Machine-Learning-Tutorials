# -*- coding: utf-8 -*-
# @File    : tut9_q2.py
# @Time    : 07/12/2022
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 2: Principal Component Analysis

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def get_cov(sdx=1., sdy=1., rotangdeg=0.):
    covar = np.array([[sdx ** 2, 0], [0, sdy ** 2]])
    rot_ang = rotangdeg / 360 * 2 * np.pi
    c = np.cos(rot_ang)
    s = np.sin(rot_ang)
    rot_mat = np.array([[c, -s], [s, c]])

    covar = np.matmul(np.matmul(rot_mat, covar), rot_mat.T)
    return covar


np.random.seed(0)
covar = get_cov(sdx=0.3, sdy=0.1, rotangdeg=23)
X = np.random.multivariate_normal([0, 0], covar, 500)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1])
ax.axis('square')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

pca = PCA(n_components=2)
pca.fit(X)

w1 = pca.components_[0]
w2 = pca.components_[1]

x_dummy = np.linspace(0, 1, 10)
ax.plot([0, w1[0]], [0, w1[1]], c='r')
ax.plot([0, w2[0]], [0, w2[1]], c='g')

plt.show()

angle = np.arctan(w1[1]/w1[0])
# it coverges to 23 when m increases or std increases
print('angle of the first principal component: ', angle*180/np.pi)
