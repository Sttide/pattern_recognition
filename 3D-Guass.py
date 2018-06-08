# -*- coding: utf-8 -*-
# Created Time    : 18-5-27 下午8:08
# Connect me with : sttide@outlook.com

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import *

fig = plt.figure()
ax = Axes3D(fig)
length = 10
step = 0.01

def build_gaussian_layer(mean, standard_deviation):
    x = np.arange(-length, length, step);
    y = np.arange(-length, length, step);
    x, y = np.meshgrid(x, y);
    z = np.exp(-((y-mean)**2 + (x - mean)**2)/(2*(standard_deviation**2)))
    z = z/(np.sqrt(2*np.pi)*standard_deviation);
    return (x, y, z);

x, y, z = build_gaussian_layer(0, 5)
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
