# -*- coding: utf-8 -*-
# @File    : main_dataset2.py
# @Time    : 10/12/2022
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tut10_tools import train_svm, plot_contourf
sys.path.append('..')
from tools import gen_sample_grid


if __name__ == '__main__':
    df = pd.read_csv('slip_data_full.csv')

    X = df.drop(columns=['m', 'slips']).values
    y = df['slips']
    standard_scaler = StandardScaler()
    standard_scaler.fit(X)
    X_standard = standard_scaler.transform(X)

    svm_mdl = train_svm(X_standard, y)

    Xgrid, x1line, x2line = gen_sample_grid(200, 200, [0, 40, 0, 360])
    f1 = 10
    f1ang = 30
    f1_arr = f1 * np.ones([len(Xgrid), 1])
    f1ang_arr = f1ang * np.ones([len(Xgrid), 1])

    Xgrid_full = np.hstack([Xgrid, f1_arr, f1ang_arr])
    Xgrid_standard = standard_scaler.transform(Xgrid_full)

    z_svm = svm_mdl.predict(Xgrid_standard).reshape(200, 200)
    _, _ = plot_contourf(x1line, x2line, z_svm, 'D: SVM')

