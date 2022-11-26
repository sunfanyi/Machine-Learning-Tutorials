# -*- coding: utf-8 -*-
# @File    : tut4_q1.py
# @Time    : 2022/11/4
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# Question 1: Scaling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('tensile_strength.csv')

t, s = df['Temperature (deg C)'], df['Ultimate tensile strength (Pa)']
t_mean, s_mean = df.mean(0)
t_std, s_std = df.std(0)  # pandas uses denominator as n-1 while numpy uses n
s_scaled = (s - s_mean) / s_std

fig, ax = plt.subplots()
plt.hist(s_scaled)
plt.show()
plt.hist(s)
plt.show()

scArray = np.array([[t_mean, s_mean], [t_std, s_std]])
np.savetxt('scaleParams.txt', scArray)
loadedScales = np.loadtxt('scaleParams.txt')
