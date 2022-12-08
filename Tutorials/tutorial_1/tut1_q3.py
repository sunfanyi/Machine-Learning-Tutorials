# -*- coding: utf-8 -*-
# @File    : tut1_q3.py
# @Time    : 2022/10/10
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 3: Generate a circular distribution

import numpy as np
import matplotlib.pyplot as plt


def cartesian_plot(r, theta, colour):
    plt.scatter(r * np.cos(theta), r * np.sin(theta), c=colour)


class1_short = np.random.normal(2, 1, 100)
class1_long = np.random.normal(8, 1, 400)
class2 = np.random.normal(5, 1, 200)

cartesian_plot(class1_short, np.random.uniform(0, 2 * np.pi, 100), 'r')
cartesian_plot(class1_long, np.random.uniform(0, 2 * np.pi, 400), 'r')
cartesian_plot(class2, np.random.uniform(0, 2 * np.pi, 200), 'b')
plt.axis('square')
plt.show()
