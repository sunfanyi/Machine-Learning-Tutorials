import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def gen_sample_grid(npx=200, npy=200, limit=1):
    x1line = np.linspace(-limit, limit, npx)
    x2line = np.linspace(-limit, limit, npy)
    x1grid, x2grid = np.meshgrid(x1line, x2line)
    Xgrid = np.array([x1grid, x2grid]).reshape([2, npx * npy]).T
    return Xgrid, x1line, x2line


# You may find lines or sections of the following code useful:
def useful_code():
    x_low = 0
    x_high = 2 * np.pi
    n_x_values = 30
    # define a set of n_x_values points between x_low and x_high (0 to
    # 2 pi in this case):
    x = np.linspace(x_low, x_high, n_x_values)

    # Normalise a vector (i.e. turn it into a unit vector):
    v = np.array([4, 8, 1])
    v_hat = v / np.linalg.norm(v)
    print("Normalised vector: ", v_hat)

    # find the minimum value within vector v:
    print("Minimum of v (should be 1): ", np.min(v))
    # find where the minimum occured - NB zero indexed
    # (NB can also do argmax() for maximum point)
    print("Minimum occurs at element: ", np.argmin(v, axis=0))

    # calculate y values for the given x values, using y = cos(x)
    y = np.cos(x)
    integral = np.trapz(y, x=x)  # integrate y using the trapezium rule
    print("Integral (should be approx 0):", integral)

    loc = np.pi  # set location to be pi (3.14...)
    # interpolate function y(x) to find value at loc - i.e. y(loc):
    val = np.interp(loc, x, y)
    print("Value at pi (should be approx -1):", val)

    # summing and squaring:
    print("Sum of y squared: ", np.sum(y ** 2))

    # do a for loop
    for cnt in range(4):
        print("Iteration: ", cnt)

    # Reshape matrix A such that it has the same shape as B:
    A = np.array([[1, 2], [3, 4]])
    print("A before reshaping:\n", A)
    B = np.array([6, 7, 8, 9])
    print("B:\n", B)
    A = np.reshape(A, B.shape)
    print("A after reshaping to match B:\n", A)

    # Multiply two matrices together:
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = np.matmul(A, B)
    print("Multiplication of \n", A, "\nand\n", B, "\ngives:\n", C)

    # Plotting functions/points:
    fig, ax = plt.subplots()  # define a figure
    # plot the points given by vectors x and y with a blue
    # solid line ('b-'). k is black, r red and g green.
    plt.plot(x, y, 'b-')
    plt.plot(x, y, 'k.')  # as above, but plot as black dots

    # 2D plotting and routines:

    # generate two random sets of values x1 and x2
    n_points = 20
    x1 = np.random.normal(loc=0, scale=1, size=n_points)
    x2 = np.random.normal(loc=0, scale=1, size=n_points)
    # Combine two vectors x1 and x2, of length n_points, into
    # a matrix of size n_points x 2
    X = np.concatenate((np.reshape(x1, [n_points, 1]),
                        np.reshape(x2, [n_points, 1])), axis=1)

    y = np.zeros([n_points])  # define y and set all values to zero
    # now put all the class values to 1 where the x1 value is
    # greater than 0.2
    y[x1 > 0.2] = 1

    # points in x and y:
    npx = 200
    npy = 200
    # generate the grid to sample 2D space:
    Xgrid, x1line, x2line = gen_sample_grid(npx, npy, 3)
    # generate an arbitrary 2D function - here do x1^2 + x2:
    z = Xgrid[:, 0] ** 2 + Xgrid[:, 1]
    # and reshape it back to the grid
    z = z.reshape([npx, npy])
    fig, ax = plt.subplots()
    # plot the values in z sampled at values given by the
    # vectors x1line, x2line:
    plt.contourf(x1line, x2line, z)
    # plot scattered values in the n_points x 2 matrix X where
    # corresponding values in the y vector equal 0:
    ax.scatter(X[y == 0, 0], X[y == 0, 1])
    # then plot where y == 1:
    ax.scatter(X[y == 1, 0], X[y == 1, 1])

# uncomment to run the code if you wish:
# useful_code()
