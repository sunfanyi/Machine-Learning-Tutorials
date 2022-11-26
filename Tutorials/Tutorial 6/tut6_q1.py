# -*- coding: utf-8 -*-
# @File    : tut6_q1.py
# @Time    : 2022/11/13
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 1: 1D function fitting

import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# ReLU function
x = np.linspace(0, 1, 200)
y = x.copy()
y[x < 0.1] = 0  # y = 0 for x < 0.1
y[x >= 0.1] = 0.5 * (x[x >= 0.1] - 0.1)  # add a gradient of +0.5 if x >= 0.1
y[x >= 0.6] = 0.25 - 0.25 / 0.2 * (x[x >= 0.6] - 0.6)
# gradient downwards from 0.6 to 0.8 (gradient calculated so that line goes from
# (0.6, 0.25) -> (0.8, 0), i.e. joins up with the other segments)
y[x > 0.8] = 0  # y = 0 for x > 0.8

fig, ax = plt.subplots()
ax.plot(x, y)

# set up a sequential neural network
model = Sequential()

# add a layer of ** nodes of ReLUs, taking a single parametric input
model.add(Dense(units=10, activation='relu', input_dim=1))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=10, activation='relu'))
# add a linear node at the end to combine the nodes together
model.add(Dense(units=1, activation='linear'))

# compile the model, trying to minimise mean squared
# error and using the Adam algorithm to fit this
model.compile(loss="mean_squared_error",
              optimizer='adam')

# fit the data provided previously, using 200 epochs and a batch size of 32
model.fit(x, y, epochs=2000, batch_size=32)

# no need to compile and train the model if the weights are set manually:
# model.layers[0].set_weights(
#     [np.array([[0.5, -(-0.5 - 0.25 / 0.2), 0.25 / 0.2]], ),
#      np.array([-0.5 * 0.1, (-0.5 - 0.25 / 0.2) * 0.6,
#                -(0.25 / 0.2) * 0.8], )])
# model.layers[1].set_weights([np.array([[1], [-1], [1]], ), np.array([0], )])

# obtain a predicted set of values from the fitted function along its length
y_pred = model.predict(x)

ax.plot(x, y_pred, 'r--')
fig.show()
