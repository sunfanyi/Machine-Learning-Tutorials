# -*- coding: utf-8 -*-
# @File    : main_dataset1.py
# @Time    : 2022/11/14
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.models import load_model
from tools import scale_features, fit_model, get_accuracy


def fit_dataset1():
    # np.random.seed(0)
    df = pd.read_csv('dataset1.csv')

    X, scaling_parameters = scale_features(np.array(df.iloc[:, 0:-1]))
    np.savetxt('fs1519-1.txt', scaling_parameters)
    y = to_categorical(df['Target hit'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=101)

    epochs = 50
    batch_size = 5
    lr = 0.0064
    num_hidden_layers = 1
    num_nodes = [6]
    activations = ['relu'] * num_hidden_layers
    model = fit_model(X_train, y_train, num_hidden_layers, num_nodes,
                      activations, epochs, batch_size, lr)
    train_acc, test_acc = get_accuracy(model, X_train, X_test,
                                       y_train, y_test, show=True)

    # model.save('dont save me fs1519-1.h5')
    return model


def fit_dataset2():
    np.random.seed(0)
    df = pd.read_csv('dataset2.csv')

    X, scaling_parameters = scale_features(np.array(df.iloc[:, 0:-1]))
    np.savetxt('fs1519-2.txt', scaling_parameters)
    y = to_categorical(df['Target hit'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=101)

    epochs = 500
    batch_size = 32
    lr = 0.006
    num_hidden_layers = 4
    num_nodes = [9, 9, 6, 9]
    activations = ['tanh'] * num_hidden_layers
    model = fit_model(X_train, y_train, num_hidden_layers, num_nodes,
                      activations, epochs, batch_size, lr)
    train_acc, test_acc = get_accuracy(model, X_train, X_test,
                                       y_train, y_test, show=True)

    # model.save('fs1519-2.h5')
    return model


if __name__ == '__main__':
    # model = fit_dataset1()
    model = fit_dataset2()

    # model = load_model('fs1519-2.h5')
    # df = pd.read_csv('dataset2.csv')
    #
    # X, scaling_parameters = scale_features(np.array(df.iloc[:, 0:-1]))
    # y = to_categorical(df['Target hit'])
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
    #                                                     random_state=101)
    # get_accuracy(model, X_train, X_test,
    #              y_train, y_test, show=True)
    #
    #