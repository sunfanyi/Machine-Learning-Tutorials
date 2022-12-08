# -*- coding: utf-8 -*-
# @File    : tut5_part3.py
# @Time    : 2022/11/12
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Part 3: Parameter optimisation with KFold on SVC

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold

sys.path.append('..')
from tools import gen_circular_distribution


def train_and_get_accuracy_KFold(X, y, C, kernel, degree=2):
    # This function train a SVC model using the input parameters and return the
    # training and test accuracy using KFold cross validation

    kf = KFold(n_splits=5, shuffle=True)

    train_acc = []  # percentage of correct classification for training error
    test_acc = []  # percentage of correct classification for test error
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        svm = SVC(C=C, gamma='auto', kernel=kernel, degree=degree)
        svm.fit(X_train, y_train)

        y_predicted_train = svm.predict(X_train)
        acc = np.sum(y_predicted_train == y_train) / len(y_train)
        train_acc.append(acc)

        y_predicted_test = svm.predict(X_test)
        acc = np.sum(y_predicted_test == y_test) / len(y_test)
        test_acc.append(acc)

    train_acc = np.mean(train_acc)
    test_acc = np.mean(test_acc)
    return train_acc, test_acc


def find_optimal_C(all_regularisations, kernel, degree=2):
    all_train_acc = []
    all_test_acc = []
    for C in all_regularisations:
        train_acc, test_acc = train_and_get_accuracy_KFold(X, y, C, kernel,
                                                           degree)
        all_train_acc.append(train_acc*100)
        all_test_acc.append(test_acc*100)

    fig, ax = plt.subplots()
    ax.plot(all_regularisations, all_train_acc, label='train_acc')
    ax.plot(all_regularisations, all_test_acc, label='test_acc')
    ax.legend()
    ax.set_xlabel('Regularisation C')
    ax.set_ylabel('Accuracy %')
    if kernel == 'poly':
        fig.suptitle('kernel = poly\ndegree = ' + str(degree))
    else:
        fig.suptitle('kernel = ' + kernel)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    X, y = gen_circular_distribution(500)
    order = np.arange(0, 10, 1)
    all_regularisations = 0.01 * 2**order
    find_optimal_C(all_regularisations, 'rbf')
    find_optimal_C(all_regularisations, 'linear')
    find_optimal_C(all_regularisations, 'poly', degree=2)

