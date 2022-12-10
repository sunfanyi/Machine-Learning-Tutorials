# -*- coding: utf-8 -*-
# @File    : main_dataset1.py
# @Time    : 10/12/2022
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import pandas as pd
from sklearn.preprocessing import StandardScaler

from tut10_tools import *
sys.path.append('..')
from tools import gen_sample_grid


if __name__ == '__main__':
    df = pd.read_csv('slip_data.csv')

    X = df.loc[:, ['f2', 'f2ang']].values
    y = df['slips']
    standard_scaler = StandardScaler()
    standard_scaler.fit(X)
    X_standard = standard_scaler.transform(X)

    nn_mdl = train_nn(X_standard, y)
    svm_mdl = train_svm(X_standard, y)

    Xgrid, x1line, x2line = gen_sample_grid(200, 200, [0, 40, 0, 360])
    Xgrid_standard = standard_scaler.transform(Xgrid)

    z_svm = svm_mdl.predict(Xgrid_standard).reshape(200, 200)
    _, _ = plot_contourf(x1line, x2line, z_svm, 'A: SVM')

    z_nn = nn_mdl.predict(Xgrid_standard)[:, 1].reshape(200, 200)
    _, _ = plot_contourf(x1line, x2line, z_nn,
                         'A: Neural Network (Probability)', colorbar=True)

    z_nn_class = (z_nn > 0.5).astype('int')
    _, _ = plot_contourf(x1line, x2line, z_nn_class,
                         'A: Neural Network (Class)')



