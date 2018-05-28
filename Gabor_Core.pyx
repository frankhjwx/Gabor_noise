from math import *
from random import *
import numpy as np

def gabor_noise(size, kernel_size, point_num, K, a, F_0, omega_0, anisotropic=True):
    img = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            img[i][j] = noise(i, j, size, K, a, F_0, kernel_size, point_num)
    return img

def gabor(K, a, F_0, omega_0, x, y):
    return K * exp(-pi * (a ** 2) * (x * x + y * y)) * cos(2*pi*F_0*(x*cos(omega_0) + y*sin(omega_0)))

def noise(x, y, size, K, a, F_0, kernel_size, point_num):
    x /= kernel_size
    y /= kernel_size
    sum = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            sum += cell(int(x)+i, int(y)+j, x - int(x) - i, y - int(y) - j, size, K, a, F_0, kernel_size, point_num)
    return sum

def cell(i, j, x, y, size, K, a, F_0, kernel_size, point_num):
    seed(i*size+j)
    sum = 0
    for i in range(point_num):
        x_i = random()
        y_i = random()
        w_i = random()*2 - 1
        omega_0_i = random()*2*pi
        sum += w_i * gabor(K, a, F_0, omega_0_i, (x-x_i)*kernel_size, (y-y_i)*kernel_size)
    return sum