# -*- coding: utf-8 -*-
# @File    : tools.py
# @Time    : 2022/11/14
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical


global n_KFold
n_KFold = 4


def scale_features(X):
    mean = np.mean(X, 0)
    std = np.std(X, 0)
    X_scaled = (X - mean) / std
    scaling_parameters = np.array([mean, std])

    return X_scaled, scaling_parameters


def get_accuracy(model, X_train, X_test, y_train, y_test, show=True):
    y_predicted_train = model.predict(X_train)
    y_predicted_train = (y_predicted_train > 0.5).astype('int')
    train_acc = np.sum(y_predicted_train == y_train) / 2 / len(y_train)

    y_predicted_test = model.predict(X_test)
    y_predicted_test = (y_predicted_test > 0.5).astype('int')
    test_acc = np.sum(y_predicted_test == y_test) / 2 / len(y_test)

    if show:
        print('Training accuracy: %.4f' % train_acc)
        print('Test accuracy: %.4f' % test_acc)
    return train_acc, test_acc


def fit_model(X, y, num_hidden_layers, num_nodes, activations, epochs,
              batch_size, lr=1e-3):
    model = Sequential()

    # hidden layers:
    model.add(Dense(units=num_nodes[0], activation=activations[0],
                    input_dim=6))
    for i in range(1, num_hidden_layers):
        model.add(Dense(units=num_nodes[i], activation=activations[i]))

    # output layer:
    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=lr),
                  metrics=['accuracy'])

    stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=20)
    model.fit(X, y, epochs=epochs, batch_size=batch_size,
              validation_split=0.2, callbacks=[stop_early])
    # model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    return model


def fit_model_KFold(X, y, num_hidden_layers, num_nodes, activations, epochs,
                    batch_size, show=True):
    kf = KFold(n_splits=n_KFold, shuffle=True)

    arr_train_acc = []  # accuracy of training set
    arr_cv_acc = []  # accuracy of corss-validation set
    for train_index, cv_index in kf.split(X):
        # further split training set with a corss valisation set
        X_train = X[train_index]
        y_train = y[train_index]
        X_cv = X[cv_index]
        y_cv = y[cv_index]

        model = fit_model(X_train, y_train, num_hidden_layers, num_nodes,
                          activations, epochs, batch_size)

        train_acc, cv_acc = get_accuracy(model, X_train, X_cv,
                                         y_train, y_cv, show=False)
        arr_train_acc.append(train_acc)
        arr_cv_acc.append(cv_acc)

    avg_train_acc = np.mean(arr_train_acc)
    avg_cv_acc = np.mean(arr_cv_acc)

    if show:
        print('Training accuracy: %.4f' % avg_train_acc)
        print('Corss-validation accuracy: %.4f' % avg_cv_acc)

    return avg_train_acc, avg_cv_acc


def optimisation_with_visualisation(all_num_hidden_layers, all_num_nodes,
                                    activation, X_train, X_test,
                                    y_train, y_test, epochs, batch_size,
                                    KFold=False):
    for i in all_num_hidden_layers:
        plot_train_acc = []
        plot_test_acc = []
        plot_num_nodes = []
        info = '%s_layer%d_KFold%d_epochs%d_batch_%d' % \
               (activation, i, n_KFold, epochs, batch_size)
        for j in all_num_nodes:
            num_nodes = [j] * int(i)  # number of nodes in each layer
            activations = [activation] * int(i)  # activations for each layer

            if KFold:
                train_acc, test_acc = fit_model_KFold(X_train, y_train, int(i),
                                                      num_nodes, activations,
                                                      epochs, batch_size,
                                                      show=False)
                test_label = 'CV accuracy'
                # path = 'model1_performance\\model_performance_KFold' \
                #        '\\%s_layer%d_KFold5.png' % (activation, i)
                path = 'model1_performance\\model_performance_KFold' \
                       '\\%s.png' % info
            else:
                model = fit_model(X_train, y_train, int(i), num_nodes,
                                  activations)
                train_acc, test_acc = get_accuracy(model, X_train, X_test,
                                                   y_train, y_test,
                                                   show=False)
                test_label = 'Test accuracy'
                path = 'model1_performance\\model_performance_wo_KFold' \
                       '\\%s_layer%d_KFold5.png' % (activation, i)

            plot_train_acc.append(train_acc)
            plot_test_acc.append(test_acc)
            plot_num_nodes.append(j)

        plt.plot(plot_num_nodes, plot_train_acc, label='Training accuracy')
        plt.plot(plot_num_nodes, plot_test_acc, label=test_label)
        plt.legend()
        plt.title(info)
        plt.xlabel('Number of nodes in each hidden layer')
        plt.ylabel('Accuracy')
        plt.savefig(path)
        plt.show()


def plot_accuracy_vs_epochs(X_train, X_test, y_train, y_test,
                            num_hidden_layers, num_nodes, activations,
                            batch_size=32):
    arr_epochs = np.arange(20, 501, 20)
    arr_train_acc, arr_test_acc = [], []
    for epochs in arr_epochs:
        model = fit_model(X_train, y_train, num_hidden_layers, num_nodes,
                          activations, epochs, batch_size)
        train_acc, test_acc = get_accuracy(model, X_train, X_test,
                                           y_train, y_test, show=False)
        # train_acc, test_acc = fit_model_KFold(X_train, y_train,
        #                                       num_hidden_layers, num_nodes,
        #                                       activations, epochs,
        #                                       batch_size, show=False)
        arr_train_acc.append(train_acc)
        arr_test_acc.append(test_acc)

    info = '%s_layer%d_%s_batch%d' % \
           (activations[0], num_hidden_layers, str(num_nodes), batch_size)
    path = 'accuracy_vs_epochs\\model1\\%s.png' % info
    fig, ax = plt.subplots()
    ax.plot(arr_epochs, arr_train_acc, label='Training accuracy')
    ax.plot(arr_epochs, arr_test_acc, label='Test accuracy')
    fig.legend()
    ax.set_title(info)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    fig.savefig(path)
    fig.show()


if __name__ == '__main__':
    # np.random.seed(0)

    df = pd.read_csv('dataset1.csv')

    X, scaling_parameters = scale_features(np.array(df.iloc[:, 0:-1]))
    y = to_categorical(df['Target hit'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=101)

    # epochs = 5
    # batch_size = 16
    # # all_num_hidden_layers = np.linspace(1, 5, 1)
    # all_num_hidden_layers = [1]
    # # all_num_nodes = np.linspace(1, 10, 10)
    # # all_num_nodes = [2, 3, 4, 5, 6, 7, 8]
    # all_num_nodes = [4]
    # # activations = ['relu', 'sigmoid', 'softplus', 'tanh']
    # activations = ['relu']
    # for activation in activations:
    #     optimisation_with_visualisation(all_num_hidden_layers, all_num_nodes,
    #                                     activation, X_train, X_test,
    #                                     y_train, y_test, epochs, batch_size,
    #                                     KFold=True)


    num_hidden_layers = 1
    num_nodes = [5] * num_hidden_layers
    activations = ['relu'] * num_hidden_layers
    plot_accuracy_vs_epochs(X, X_test, y, y_test,
                            num_hidden_layers, num_nodes, activations,
                            batch_size=16)

    # train_acc, cv_acc = fit_model_KFold(X, y, num_hidden_layers, num_nodes,
    #                                     activations, show=True)
    #
    # model = fit_model(X_train, y_train, num_hidden_layers, num_nodes,
    #                   activations)
    # train_acc, test_acc = get_accuracy(model, X_train, X_test,
    #                                    y_train, y_test, show=True)
