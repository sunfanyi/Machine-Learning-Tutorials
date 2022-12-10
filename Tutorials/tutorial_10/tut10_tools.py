# -*- coding: utf-8 -*-
# @File    : tut10_tools.py
# @Time    : 10/12/2022
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical


def train_svm(X, y):
    svm_mdl = SVC(C=1000, gamma='scale', kernel='rbf')
    svm_mdl.fit(X, y)
    return svm_mdl


def train_nn(X, y):
    y_binary = to_categorical(y)
    nn_mdl = Sequential()
    nn_mdl.add(Dense(units=4, activation='tanh', input_dim=2))
    nn_mdl.add(Dense(units=2, activation='softmax'))
    nn_mdl.compile(loss='categorical_crossentropy',
                  optimizer='sgd')
    nn_mdl.fit(X, y_binary, epochs=1200, batch_size=32)
    return nn_mdl


def plot_contourf(x1line, x2line, z, title, colorbar=False):
    fig, ax = plt.subplots()
    contourf_ = ax.contourf(x1line, x2line, z)
    if colorbar:
        fig.colorbar(contourf_)
    ax.set_xlabel('f2')
    ax.set_ylabel('f2ang')
    ax.set_title(title)
    fig.show()

    # ax.plot(X[y == 0, 0], X[y == 0, 1], 'rx')
    # ax.plot(X[y == 1, 0], X[y == 1, 1], 'bx')
    # fig.show()

    return ax, fig
