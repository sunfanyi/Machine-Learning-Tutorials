# -*- coding: utf-8 -*-
"""tutorial7_ph_2022.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-66gaBiA7-ZV7FW7ZTOCDCPgTjfI0kSZ

Part 1
"""

import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


def gen_gaussian(x, mu, sig):
    return 1 / np.sqrt(2 * np.pi) / sig * np.exp(-((x - mu) / sig) ** 2 / 2)


mu = 0
sigma = 1
X = np.random.normal(loc=mu, scale=sigma, size=[40, 1])

kde = KernelDensity(kernel='tophat', bandwidth=1).fit(X)

x_sample = np.linspace(-5, 5, 1000)
p = np.exp(kde.score_samples(x_sample.reshape(len(x_sample), -1)))

fig, ax = plt.subplots()
plt.plot(x_sample, p)

kde = KernelDensity(kernel='tophat', bandwidth=0.2).fit(X)
p = np.exp(kde.score_samples(x_sample.reshape(len(x_sample), -1)))
plt.plot(x_sample, p)

kde = KernelDensity(kernel='tophat', bandwidth=4.0).fit(X)
p = np.exp(kde.score_samples(x_sample.reshape(len(x_sample), -1)))
plt.plot(x_sample, p)

p_true = gen_gaussian(x=x_sample, mu=mu, sig=sigma)
plt.plot(x_sample, p_true, 'k--')

kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(X)

x_sample = np.linspace(-5, 5, 1000)
p = np.exp(kde.score_samples(x_sample.reshape(len(x_sample), -1)))

fig, ax = plt.subplots()
plt.plot(x_sample, p)

kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
p = np.exp(kde.score_samples(x_sample.reshape(len(x_sample), -1)))
plt.plot(x_sample, p)

kde = KernelDensity(kernel='gaussian', bandwidth=4.0).fit(X)
p = np.exp(kde.score_samples(x_sample.reshape(len(x_sample), -1)))
plt.plot(x_sample, p)

p_true = gen_gaussian(x=x_sample, mu=mu, sig=sigma)
plt.plot(x_sample, p_true, 'k--')

"""Part 2"""

X_vis = np.array([[0]])
kde = KernelDensity(kernel='exponential', bandwidth=0.5).fit(X_vis)
p_vis = np.exp(kde.score_samples(x_sample.reshape(len(x_sample), -1)))
plt.plot(x_sample, p_vis)

"""Part 3"""


def gen_circular_distribution(n=500, scale=1):
    a = np.round(n / 7).astype('int')
    b = np.round(2 * n / 7).astype('int')
    c = n - a - b
    r1 = np.concatenate(
        [np.random.normal(loc=2, scale=scale, size=[a, 1]),
         np.random.normal(loc=8, scale=scale, size=[c, 1])])
    r2 = np.random.normal(loc=5, scale=scale, size=[b, 1])

    th1 = np.random.uniform(low=0, high=2 * np.pi, size=[a + c, 1])
    th2 = np.random.uniform(low=0, high=2 * np.pi, size=[b, 1])

    x1a = r1 * np.cos(th1)
    x2a = r1 * np.sin(th1)

    x1b = r2 * np.cos(th2)
    x2b = r2 * np.sin(th2)

    X = np.concatenate(
        [np.concatenate([x1a.reshape([a + c, 1]), x1b.reshape([b, 1])]),
         np.concatenate([x2a.reshape([a + c, 1]), x2b.reshape([b, 1])])],
        axis=1)

    y = np.concatenate([np.zeros([a + c, 1]), np.ones([b, 1])]).squeeze()
    return X, y


def gen_sample_grid(npx=200, npy=200, limit=1):
    x1line = np.linspace(-limit, limit, npx)
    x2line = np.linspace(-limit, limit, npy)
    x1grid, x2grid = np.meshgrid(x1line, x2line)
    Xgrid = np.array([x1grid, x2grid]).reshape([2, npx * npy]).T
    return Xgrid, x1line, x2line


X, y = gen_circular_distribution(200)
X1 = X[y == 0, :]
X2 = X[y == 1, :]

b = 1
# b = 0.2
kde1 = KernelDensity(kernel='gaussian', bandwidth=b).fit(X1)
kde2 = KernelDensity(kernel='gaussian', bandwidth=b).fit(X2)

Xgrid, x1line, x2line = gen_sample_grid(200, 200, 10)

p1_grid = np.exp(kde1.score_samples(Xgrid)).reshape(200, 200)
p2_grid = np.exp(kde2.score_samples(Xgrid)).reshape(200, 200)

state_of_nature = np.zeros([200, 200])
state_of_nature[p1_grid < p2_grid] = 1

fig, ax = plt.subplots()
plt.contourf(x1line, x2line, state_of_nature)

"""Part 4"""

from sklearn.neighbors import KNeighborsClassifier

near = 1
neigh = KNeighborsClassifier(n_neighbors=near)
neigh.fit(X, y)

son = neigh.predict(Xgrid).reshape(200, 200)
fig, ax = plt.subplots()
plt.contourf(x1line, x2line, son)
ax.scatter(X[y == 0, 0], X[y == 0, 1])
ax.scatter(X[y == 1, 0], X[y == 1, 1])

"""Part 5"""

# define the number of pixels in each direction in the final image:
# nx = 20
nx = 40
# nx = 10

# define the maximum value in x and y
r = 10

# make a single vector which spans the range with the right number of points
gridDim = np.linspace(-r, r, nx)

# get the spacing of the vector
dx = gridDim[1] - gridDim[0]
# get the lowest value in the vector
minx = np.min(gridDim)

# generate the distribution
np.random.seed(0)
X, y = gen_circular_distribution(2000)

# just take a single component from the distribution
Xuse = X[y == 1, :]

# convert the continuous values of x and y coordinates in Xuse[:, 0] and Xuse[:, 1]
# into discrete integers, by subtracting off the minimum and dividing by the spacing,
# then rounding and converting into integers
Xgrid = (np.round((Xuse[:, 0] - minx) / dx)).astype('int')
Ygrid = (np.round((Xuse[:, 1] - minx) / dx)).astype('int')

# set up a grid of density values which will contain the final image - set to 0 initially
density = np.zeros([nx, nx])

# loop through each value...
for i in range(len(Xgrid)):
    # Add 1 to the density for the corresponding cell, identified by the rounding approach above
    density[Xgrid[i], Ygrid[i]] += 1

# normalise it all
totDen = np.sum(density[:])
density /= totDen * dx * dx

# plot it
fig, ax = plt.subplots()
plt.contourf(gridDim, gridDim, density.T)

ax.scatter(X[y == 1, 0], X[y == 1, 1], marker='x', s=2)
plt.colorbar()
plt.xlim(-r, r)
plt.ylim(-r, r)
plt.show()
