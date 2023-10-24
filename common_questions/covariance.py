# -*- coding: utf-8 -*-
# @File    : covariance.py
# @Time    : 12/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import numpy as np

X = np.array([[1, -3],
              [0, 1],
              [2, 1]])

print(np.cov(X.T, bias=True))

# alternatively:
mu = np.mean(X, axis=0)
cov = np.matmul((X-mu).T, (X-mu)) / 3
print(cov)
