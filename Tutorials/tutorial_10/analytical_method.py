# -*- coding: utf-8 -*-
# @File    : analytical_method.py
# @Time    : 10/12/2022
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question C: Using analytical method to identify the slip

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('..')
from tools import gen_sample_grid


if __name__ == '__main__':
    df = pd.read_csv('slip_data.csv')
    X = df.loc[:, ['f2', 'f2ang']].values
    y = df['slips']

    m = 3
    f1 = 18
    f1ang = 20
    # f1 = 10
    # f1ang = 30

    # with f2 = [0:40], f2ang = [0:360]
    Xgrid, x1line, x2line = gen_sample_grid(200, 200, [0, 40, 0, 360])
    f2 = Xgrid[:, 0]
    f2ang = Xgrid[:, 1]

    F_h = f1 * np.cos(f1ang * np.pi / 180) + f2 * np.cos(f2ang * np.pi / 180)
    F_v = f1 * np.sin(f1ang * np.pi / 180) + f2 * np.sin(f2ang * np.pi / 180)
    F_res = np.sqrt(F_h ** 2 + F_v ** 2)

    mu = 0.5
    g = 9.81
    f = mu * m * g

    slip_info = (F_res > f).reshape(200, 200)

    fig, ax = plt.subplots()
    ax.contourf(x1line, x2line, slip_info)
    ax.set_xlabel('f2')
    ax.set_ylabel('f2ang')
    ax.set_title('C: Analytical Method')
    fig.show()
