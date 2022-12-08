# -*- coding: utf-8 -*-
# @File    : tut3_q3.py
# @Time    : 2022/10/24
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 3: Using Scikit-learn to perform regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures

# read in the CSV file
df = pd.read_csv('hdpeVel.csv')
# set the 'index' column as the one containing the temperature values
df = df.set_index('T/C f/MHz')

# extract the frequency values (and scale since they are MHz)
freq = df.columns.values.astype(float) * 1e6  # f/MHz
# extract the temperature values
temp = df.index.values.astype(float)  # T/C

# extract the main part - the velocity values
vel = df.to_numpy()
# calculate the total number of values
tot_values = len(freq) * len(temp)

x1grid, x2grid = np.meshgrid(freq, temp)
Xgrid = np.concatenate([x1grid.reshape([tot_values, 1]),
                        x2grid.reshape([tot_values, 1])], axis=1)
ygrid = vel.reshape([tot_values, 1])

reg = LinearRegression()
reg.fit(Xgrid, ygrid)

y_lin = reg.predict(Xgrid)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xgrid[:, 0], Xgrid[:, 1], ygrid, marker='x', color='#000000')
ax.scatter(Xgrid[:, 0], Xgrid[:, 1], y_lin, marker='o', color='#ff0000')
plt.show()


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(Xgrid)

print(X_poly.shape)
print(poly.powers_)


reg_poly = LinearRegression()
reg_poly.fit(X_poly, ygrid)

