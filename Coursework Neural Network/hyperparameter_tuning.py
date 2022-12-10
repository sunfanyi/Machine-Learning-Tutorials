# -*- coding: utf-8 -*-
# @File    : hyperparameter_tuning.py
# @Time    : 2022/11/16
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.python.keras.utils.np_utils import to_categorical

from nn_tools import scale_features, get_accuracy


# model = Sequential()
#
# # hidden layers:
# model.add(Dense(units=num_nodes[0], activation=activations[0],
#                 input_dim=6))
# for i in range(1, num_hidden_layers):
#     model.add(Dense(units=num_nodes[i], activation=activations[i]))
#
# # output layer:
# model.add(Dense(units=2, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd')
#
# model.fit(X, y, epochs=epochs, batch_size=batch_size)


def build_model(hp):
    model = Sequential()

    model.add(layers.Flatten())
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 3, 5)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=1, max_value=10, step=1),
                # activation=hp.Choice("activation", ["relu", "tanh", "sigmoid"]),
                activation='tanh',
            )
        )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))

    # # hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    # hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=10, step=1)
    # hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=10, step=1)
    # learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2,
    #                          sampling="log")
    #
    # model.add(Dense(units=hp_layer_1, activation='relu'), input_dim=6)
    # # model.add(Dense(units=hp_layer_1, activation='relu'))
    # # model.add(Dense(units=hp_layer_2, activation='relu'))
    model.add(layers.Dense(units=2, activation='softmax'))

    # learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2,
    #                          sampling="log")
    learning_rate = hp.Float("lr", min_value=1e-3, max_value=1e-2,
                             sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


build_model(kt.HyperParameters())

tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=2000,
                     factor=3,
                     directory='dir2',
                     project_name='dataset2_5')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)

df = pd.read_csv('dataset2.csv')

X, scaling_parameters = scale_features(np.array(df.iloc[:, 0:-1]))
np.savetxt('fs1519-2.txt', scaling_parameters)
y = to_categorical(df['Target hit'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=101)

tuner.search(X_train, y_train, epochs=250, validation_split=0.2,
             callbacks=[stop_early])

tuner.results_summary()
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
# history = model.fit(X_train, y_train, epochs=250, validation_split=0.2,
#                     callbacks=[stop_early])

# model2 = model
# stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# model2.fit(X_train, y_train, epochs=250, validation_split=0.2,
#            callbacks=[stop_early])
# get_accuracy(model2, X_train, X_test, y_train, y_test, show=True)

# model2 = model
# model2.compile(loss='categorical_crossentropy',
#               optimizer=tf.keras.optimizers.Adam(learning_rate=0.0029),
#               metrics=['accuracy'])
# model2.fit(X_train, y_train, epochs=500, batch_size=4, validation_split=0.2)
# get_accuracy(model2, X_train, X_test, y_train, y_test, show=True)
