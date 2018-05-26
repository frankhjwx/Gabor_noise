from random import *
import math
import cv2
from FFT import *


class Sparse_Convolution_Noise:
    def __init__(self, width=512, height=512, grid_size=40, point_num=3):
        self.img = np.ones([width, height])
        self.img /= 2
        self.width = width
        self.height = height
        width_num = int(math.floor(width / 2 / grid_size))
        height_num = int(math.floor(height / 2 / grid_size))
        noise = []
        for i in range(-width_num, width_num + 1):
            for j in range(-height_num, height_num + 1):
                x = i * grid_size + width / 2 - grid_size / 2
                y = j * grid_size + height / 2 - grid_size / 2
                for k in range(point_num):
                    dx = randint(0, grid_size)
                    dy = randint(0, grid_size)
                    while not self.valid_position(x+dx, y+dy):
                        dx = randint(0, grid_size)
                        dy = randint(0, grid_size)
                    weight = random() * 2 - 1
                    noise.append({'x': int(x+dx), 'y': int(y+dy), 'w': weight})
        for n in noise:
            self.img[n['x'], n['y']] = n['w'] * 0.5 + 0.5
        self.noise = noise

    def valid_position(self, x, y):
        if x < 0 or x >= self.width:
            return False
        if y < 0 or y >= self.height:
            return False
        return True

    def spacial_display(self):
        cv2.namedWindow('Sparse_Convolution_Noise_spacial', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Sparse_Convolution_Noise_spacial', self.img)
        cv2.waitKey(0)

    def frequency_display(self):
        cv2.namedWindow('Sparse_Convolution_Noise_frequency', cv2.WINDOW_AUTOSIZE)
        img_frequency = spatial_to_frequency(self.img)
        cv2.imshow('Sparse_Convolution_Noise_frequency', img_frequency)
        cv2.waitKey(0)

    def frequency_simulate_display(self):
        img_frequency = np.zeros([self.width, self.height])
        cv2.namedWindow('Sparse_Convolution_Noise_simulate_frequency', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Sparse_Convolution_Noise_simulate_frequency', img_frequency)
        cv2.waitKey(0)

if __name__ == '__main__':
    scn = Sparse_Convolution_Noise()
    scn.spacial_display()
    scn.frequency_display()
    scn.frequency_simulate_display()

