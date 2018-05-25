from Gaussian_Kernel import *
from Harmonic_Kernel import *
import numpy as np
import cv2
from FFT import *

class Gabor_Kernel:
    def __init__(self, img_size=256, K=1, a=0.06, F_0=0.15, omega_0=0.7):
        self.size = img_size
        self.a = a * img_size
        self.K = K
        self.F0 = img_size * F_0
        self.omega_0 = omega_0
        gaussian = Gaussian_Kernel(img_size, K, a)
        harmonic = Harmonic_Kernel(img_size, F_0, omega_0)
        self.img = np.multiply(gaussian.img, harmonic.img)
        print(self.img)

    def spacial_display(self):
        cv2.namedWindow('Gabor_Kernel_spacial', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Gabor_Kernel_spacial', self.img)
        cv2.waitKey(0)

    def frequency_display(self):
        cv2.namedWindow('Gaussian_Kernel_frequency', cv2.WINDOW_AUTOSIZE)
        img_frequency = spatial_to_frequency(self.img)
        cv2.imshow('Gaussian_Kernel_frequency', img_frequency)
        cv2.waitKey(0)

    # 理想状态下的频域图
    def frequency_simulate_display(self):
        img_frequency = np.zeros([self.size, self.size])
        dx = int(self.F0 * sin(self.omega_0))
        dy = int(self.F0 * cos(self.omega_0))
        half_size = int(self.size / 2)
        c1 = [half_size + dx, half_size + dy]
        c2 = [half_size - dx, half_size - dy]
        for i in range(self.size):
            for j in range(self.size):
                x1 = (i - c1[0]) / self.size
                y1 = (j - c1[1]) / self.size
                x2 = (i - c2[0]) / self.size
                y2 = (j - c2[1]) / self.size
                value = self.K / (2 * (self.a/self.size)**2) * exp(-pi / ((self.a/self.size)**2) * (x1*x1 + y1*y1)) + \
                        self.K / (2 * (self.a/self.size) ** 2) * exp(-pi / ((self.a/self.size) ** 2) * (x2*x2 + y2*y2))
                img_frequency[i][j] = value

        cv2.namedWindow('Gaussian_Kernel_simulate_frequency', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Gaussian_Kernel_simulate_frequency', img_frequency)
        cv2.waitKey(0)


if __name__ == '__main__':
    gabor = Gabor_Kernel(img_size=256, K=1, a=0.06, F_0=0.15, omega_0=0.2)
    gabor.spacial_display()
    gabor.frequency_display()
    gabor.frequency_simulate_display()
