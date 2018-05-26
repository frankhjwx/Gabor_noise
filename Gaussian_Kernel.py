from math import *
import cv2
from FFT import *


class Gaussian_Kernel:
    def __init__(self, img_size=256, K=1, a=0.06):
        self.size = img_size
        self.a = a * img_size
        self.K = K
        self.img = np.zeros([self.size, self.size])
        for i in range(self.size):
            for j in range(self.size):
                x = (i-self.size/2)/self.size
                y = (j-self.size/2)/self.size
                value = self.K * exp(-pi * (self.a**2) * (x*x + y*y))
                self.img[i][j] = value

    def spacial_display(self):
        cv2.namedWindow('Gaussian_Kernel_spacial', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Gaussian_Kernel_spacial', self.img)
        cv2.waitKey(0)

    def frequency_display(self):
        cv2.namedWindow('Gaussian_Kernel_frequency', cv2.WINDOW_AUTOSIZE)
        img_frequency = spatial_to_frequency(self.img)
        cv2.imshow('Gaussian_Kernel_frequency', img_frequency)
        cv2.waitKey(0)

    # 理想状态下的频域图
    def frequency_simulate_display(self):
        img_frequency = np.zeros([self.size, self.size])
        for i in range(self.size):
            for j in range(self.size):
                x = (i-self.size/2)/self.size
                y = (j-self.size/2)/self.size
                value = self.K * exp(-pi * ((self.size / 2 / self.a) ** 2) * (x * x + y * y))
                img_frequency[i][j] = value

        cv2.namedWindow('Gaussian_Kernel_simulate_frequency', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Gaussian_Kernel_simulate_frequency', img_frequency)
        cv2.waitKey(0)


if __name__ == '__main__':
    gs = Gaussian_Kernel(img_size=256, K=5, a=0.1)
    gs.spacial_display()
    gs.frequency_display()
    gs.frequency_simulate_display()
