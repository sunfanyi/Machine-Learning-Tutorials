# -*- coding: utf-8 -*-
# @File    : tut6_q2.py
# @Time    : 2022/11/14
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 2: Two-parameter example

import sys
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical

sys.path.append('..')
from tools import gen_simple_circular_distribution

np.random.seed(0)
X, y = gen_simple_circular_distribution(200)
y_binary = to_categorical(y)

model = Sequential()

model.add(Dense(units=4, activation='relu', input_dim=2))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd')

model.fit(X, y_binary, epochs=250, batch_size=32)

x1line = np.linspace(-10, 10, 200)
x2line = np.linspace(-10, 10, 200)
x1grid, x2grid = np.meshgrid(x1line, x2line)
Xgrid = np.array([x1grid, x2grid]).reshape([2, 200*200]).T

# use either y_predicted[:, 0] or y_predicted[:, 1] as they represent the
# probability of the points of being each class, e.g., y_predicted[i, 0] = 0.93
# is equivalent to y_predicted[i, 1] = 0.07
Z = model.predict(Xgrid)[:, 0].reshape([200, 200])
# Z = model.predict(Xgrid)[:, 1].reshape([200, 200])

fig, ax = plt.subplots()
contourf_ = ax.contourf(x1line, x2line, Z)
# Values on colorbar represent the probability of the points of being the
# corresponding classes
fig.colorbar(contourf_)
ax.plot(X[y == 0, 0], X[y == 0, 1], 'rx')
ax.plot(X[y == 1, 0], X[y == 1, 1], 'gx')
fig.suptitle('Training data')
fig.show()

model.save('model.h5')
