from Gabor_Kernel import *
from Sparse_Convolution_Noise import *
import scipy.ndimage


class Gabor_Noise:
    def __init__(self, size=512, grid_size=50, point_num=64, K=1, a=0.06, F_0=0.1, omega_0=0.7, anisotropic=True):
        self.size = size
        self.grid_size = grid_size
        self.a = a
        self.K = K
        self.F_0 = F_0
        self.omega_0 = omega_0
        self.point_num = point_num
        scn = Sparse_Convolution_Noise(width=size, height=size, grid_size=grid_size, point_num=point_num)
        gabor_kernel = Gabor_Kernel(img_size=size, K=K, a=a, F_0=F_0, omega_0=omega_0)
        if anisotropic:
            self.img = cv2.filter2D(scn.img, -1, gabor_kernel.img)
        else:
            self.img = np.zeros([size, size])
            # tbd
            for i in range(self.size):
                for j in range(self.size):
                    print(i,j)
                    self.img[i][j] = self.noise(i, j)

    def gabor(self, K, a, F_0, omega_0, x, y):
        return K * exp(-pi * (a ** 2) * (x * x + y * y)) * cos(2*pi*F_0*(x*cos(omega_0) + y*sin(omega_0)))

    def noise(self, x, y):
        x /= self.grid_size
        y /= self.grid_size
        sum = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                sum += self.cell(int(x)+i, int(y)+j, x - int(x) - i, y - int(y) - j)
        return sum

    def cell(self, i, j, x, y):
        seed(i*self.size+j)
        sum = 0
        for i in range(self.point_num):
            x_i = random()
            y_i = random()
            w_i = random()*2 - 1
            omega_0_i = random()*2*pi
            sum += w_i * self.gabor(self.K, self.a, self.F_0, omega_0_i, (x-x_i)*self.grid_size, (y-y_i)*self.grid_size)
        return sum

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
        for i in range(self.size):
            for j in range(self.size):
                x1 = (i - c1[0]) / self.size
                y1 = (j - c1[1]) / self.size
                x2 = (i - c2[0]) / self.size
                y2 = (j - c2[1]) / self.size
                value = self.K * exp(-pi * ((self.size / 2 / self.a) ** 2) * (x1 * x1 + y1 * y1)) + \
                        self.K * exp(-pi * ((self.size / 2 / self.a) ** 2) * (x2 * x2 + y2 * y2))
                img_frequency[i][j] = value

        cv2.namedWindow('Gabor_Kernel_simulate_frequency', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Gabor_Kernel_simulate_frequency', img_frequency)
        cv2.waitKey(0)


if __name__ == '__main__':
    #gabor = Gabor_Noise(size=128, grid_size=50, point_num=16, anisotropic=False)
    gabor = Gabor_Noise(point_num=16, anisotropic=False)
    gabor.spacial_display()
    gabor.frequency_display()
    gabor.frequency_simulate_display()

