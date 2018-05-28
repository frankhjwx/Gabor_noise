from Gabor_Kernel import *
from Sparse_Convolution_Noise import *
import Gabor_Core

class Gabor_Noise:
    def __init__(self, size=512, grid_size=50, point_num=64, K=1, a=0.06, F_0=0.1, omega_0=0.7, anisotropic=True):
        self.size = size
        self.grid_size = grid_size
        self.a = a
        self.K = K
        self.F_0 = F_0
        self.omega_0 = omega_0
        self.point_num = point_num
        self.anisotropic = anisotropic
        scn = Sparse_Convolution_Noise(width=size, height=size, grid_size=grid_size, point_num=point_num)
        gabor_kernel = Gabor_Kernel(img_size=size, K=K, a=a, F_0=F_0, omega_0=omega_0)
        if anisotropic:
            self.img = cv2.filter2D(scn.img, -1, gabor_kernel.img)
        else:
            self.img = np.zeros([size, size])
            # tbd
            self.img = Gabor_Core.gabor_noise(self.size, self.grid_size, self.point_num, self.K,
                                              self.a, self.F_0, self.omega_0, anisotropic)
            print(self.img)

    def spacial_display(self):
        cv2.namedWindow('Gabor_Noise_spacial', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Gabor_Noise_spacial', self.img/2+0.5)
        cv2.waitKey(0)

    def frequency_display(self):
        cv2.namedWindow('Gabor_Noise_frequency', cv2.WINDOW_AUTOSIZE)
        img_frequency = spatial_to_frequency(self.img)
        cv2.imshow('Gabor_Noise_frequency', img_frequency)
        cv2.waitKey(0)

    def frequency_simulate_display(self):
        img_frequency = np.zeros([self.size, self.size])
        self.a *= self.size
        self.F_0 *= self.size
        dx = int(self.F_0 * sin(self.omega_0))
        dy = int(self.F_0 * cos(self.omega_0))
        half_size = int(self.size / 2)
        c1 = [half_size + dx, half_size + dy]
        c2 = [half_size - dx, half_size - dy]
        d2 = (dx**2+dy**2)/(self.size**2)
        if self.anisotropic:
            for i in range(self.size):
                for j in range(self.size):
                    x1 = (i - c1[0]) / self.size
                    y1 = (j - c1[1]) / self.size
                    x2 = (i - c2[0]) / self.size
                    y2 = (j - c2[1]) / self.size
                    value = self.K * exp(-pi * ((self.size / 2 / self.a) ** 2) * (x1 * x1 + y1 * y1)) + \
                            self.K * exp(-pi * ((self.size / 2 / self.a) ** 2) * (x2 * x2 + y2 * y2))
                    img_frequency[i][j] = value
        else:
            for i in range(self.size):
                for j in range(self.size):
                    d1 = ((i-half_size)**2 + (j-half_size)**2) / (self.size**2)
                    d = abs(d1-d2)
                    if d < 0.002:
                        d = 0.002
                    value = self.K * exp(-pi * ((self.size / 2 / self.a) ** 2) * d)
                    img_frequency[i][j] = value

        cv2.namedWindow('Gabor_Kernel_simulate_frequency', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Gabor_Kernel_simulate_frequency', img_frequency)
        cv2.waitKey(0)



if __name__ == '__main__':
    #gabor = Gabor_Noise(size=128, grid_size=50, point_num=16, anisotropic=False)
    gabor = Gabor_Noise(size=512, point_num=16, anisotropic=True)
    gabor.spacial_display()
    gabor.frequency_display()
    gabor.frequency_simulate_display()
