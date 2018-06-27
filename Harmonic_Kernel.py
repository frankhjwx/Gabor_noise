from math import *
import cv2
from FFT import *


class Harmonic_Kernel:
    def __init__(self, img_size=256, F_0=0.15, omega_0=0.7):
        self.size = img_size
        self.img = np.zeros([self.size, self.size])
        self.F0 = img_size * F_0
        self.omega_0 = omega_0
        for i in range(self.size):
            for j in range(self.size):
                x = (j - self.size / 2) / self.size
                y = (i - self.size / 2) / self.size
                value = cos(2*pi*self.F0*(x*cos(self.omega_0) + y*sin(self.omega_0)))
                self.img[i][j] = value

    def spacial_display(self):
        cv2.namedWindow('Harmonic_Kernel_spacial', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Harmonic_Kernel_spacial', self.img)
        cv2.waitKey(0)

    def frequency_display(self):
        cv2.namedWindow('Harmonic_Kernel_frequency', cv2.WINDOW_AUTOSIZE)
        img_frequency = spatial_to_frequency(self.img)
        cv2.imshow('Harmonic_Kernel_frequency', img_frequency)
        cv2.waitKey(0)

    # 理想状态下的频域图
    def frequency_simulate_display(self):
        img_frequency = np.zeros([self.size, self.size])
        dx = int(self.F0 * sin(self.omega_0))
        dy = int(self.F0 * cos(self.omega_0))
        half_size = int(self.size/2)
        img_frequency[half_size + dx, half_size + dy] = 1
        img_frequency[half_size - dx, half_size - dy] = 1

        cv2.namedWindow('Harmonic_Kernel_simulate_frequency', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Harmonic_Kernel_simulate_frequency', img_frequency)
        cv2.waitKey(0)


if __name__ == '__main__':
    hk = Harmonic_Kernel(img_size=512, F_0=0.06, omega_0=0.2)
    hk.spacial_display()
    hk.frequency_display()
    hk.frequency_simulate_display()
