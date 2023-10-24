# -*- coding: utf-8 -*-
# @File    : tut2_q2.py
# @Time    : 2022/10/17
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 2: Classification with Bayes

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def train_model(X_train, y_train, Xgrid):
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    classVals = clf.predict(Xgrid)
    classGrid = np.reshape(classVals, [npx, npy])
    plt.contourf(x1line, x2line, classGrid)
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1])
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1])
    plt.axis('square')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title("training data")
    plt.show()

    return clf


def test_model(clf, X_test, y_test):
    y_test_model = clf.predict(X_test)

    nTot = len(y_test)
    nMatch = 0
    for i in range(nTot):
        if y_test[i] == y_test_model[i]:
            nMatch += 1

    accuracy = nMatch / nTot
    print("%.2f%%" % (100 * accuracy))
    return accuracy


def visualise_proba(clf, Xgrid, npx, npy, x1line, x2line):
    probVals = clf.predict_proba(Xgrid)
    probGrid = np.reshape(probVals[:, 0], [npx, npy])
    plt.contourf(x1line, x2line, probGrid)
    plt.axis('square')
    plt.title("test data")
    plt.show()


if __name__ == '__main__':
    np.random.seed(5)
    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=2,
                                        n_informative=2,
                                        n_redundant=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    npx = 200
    npy = 200
    x1line = np.linspace(-3, 3, npx)
    x2line = np.linspace(-3, 3, npy)
    x1grid, x2grid = np.meshgrid(x1line, x2line)
    Xgrid = np.array([x1grid, x2grid]).reshape([2, npx * npy]).T

    clf = train_model(X_train, y_train, Xgrid)
    accuracy = test_model(clf, X_test, y_test)
    visualise_proba(clf, Xgrid, npx, npy, x1line, x2line)
